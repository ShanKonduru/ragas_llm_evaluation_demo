import os
import platform
import subprocess
import re

def get_windows_proxy_from_registry():
    """
    Attempts to read proxy settings from the Windows Registry (Internet Explorer/Edge settings).
    Returns (http_proxy, https_proxy) or (None, None).
    """
    try:
        # Import winreg only on Windows
        import winreg

        # Path to the Internet Settings key
        reg_path = r"Software\Microsoft\Windows\CurrentVersion\Internet Settings"

        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, reg_path) as key:
            # Check if proxy is enabled
            proxy_enable, _ = winreg.QueryValueEx(key, "ProxyEnable")
            if proxy_enable == 0:
                return None, None # Proxy is not enabled

            proxy_server, _ = winreg.QueryValueEx(key, "ProxyServer")
            proxy_override, _ = winreg.QueryValueEx(key, "ProxyOverride")

            http_proxy = None
            https_proxy = None

            # The ProxyServer value typically contains both HTTP and HTTPS proxies,
            # often separated by semicolons, or just a single address.
            # Example: "http=192.168.1.100:8080;https=192.168.1.100:8080"
            # Or: "192.168.1.100:8080" (for both)

            if proxy_server:
                # Try to parse specific http/https entries first
                http_match = re.search(r'http=([^;]+)', proxy_server, re.IGNORECASE)
                https_match = re.search(r'https=([^;]+)', proxy_server, re.IGNORECASE)

                if http_match:
                    http_proxy = http_match.group(1)
                    if not http_proxy.startswith(('http://', 'https://')):
                        http_proxy = f"http://{http_proxy}" # Assume http if not specified
                if https_match:
                    https_proxy = https_match.group(1)
                    if not https_proxy.startswith(('http://', 'https://')):
                        https_proxy = f"http://{https_proxy}" # Assume http if not specified

                # If no specific http/https, assume the whole string is the proxy for both
                if not http_proxy and not https_proxy:
                    if not proxy_server.startswith(('http://', 'https://')):
                        proxy_server = f"http://{proxy_server}" # Assume http if not specified
                    http_proxy = proxy_server
                    https_proxy = proxy_server

            return http_proxy, https_proxy

    except ImportError:
        # winreg not available (not on Windows)
        return None, None
    except FileNotFoundError:
        # Registry key not found
        return None, None
    except Exception as e:
        print(f"Error reading Windows Registry proxy: {e}")
        return None, None

def get_macos_proxy_from_scutil():
    """
    Attempts to read proxy settings from macOS using scutil.
    Returns (http_proxy, https_proxy) or (None, None).
    """
    http_proxy = None
    https_proxy = None
    try:
        # Query active HTTP proxy
        result = subprocess.run(['scutil', '--proxy'], capture_output=True, text=True, check=True)
        output = result.stdout

        # Example output for HTTP proxy:
        # <dictionary> {
        #   HTTPEnable : 1
        #   HTTPPort : 8080
        #   HTTPProxy : 192.168.1.1
        # }
        # Example output for HTTPS proxy:
        # <dictionary> {
        #   HTTPSEnable : 1
        #   HTTPSPort : 8080
        #   HTTPSProxy : 192.168.1.1
        # }

        http_enable = re.search(r'HTTPEnable : (\d+)', output)
        if http_enable and http_enable.group(1) == '1':
            http_proxy_host = re.search(r'HTTPProxy : (.+)', output)
            http_proxy_port = re.search(r'HTTPPort : (\d+)', output)
            if http_proxy_host and http_proxy_port:
                http_proxy = f"http://{http_proxy_host.group(1)}:{http_proxy_port.group(1)}"

        https_enable = re.search(r'HTTPSEnable : (\d+)', output)
        if https_enable and https_enable.group(1) == '1':
            https_proxy_host = re.search(r'HTTPSProxy : (.+)', output)
            https_proxy_port = re.search(r'HTTPSPort : (\d+)', output)
            if https_proxy_host and https_proxy_port:
                https_proxy = f"http://{https_proxy_host.group(1)}:{https_proxy_port.group(1)}" # macOS typically uses HTTP for HTTPS proxy

    except FileNotFoundError:
        print("scutil not found. (Not on macOS or path issue)")
    except subprocess.CalledProcessError as e:
        print(f"Error running scutil: {e}")
    except Exception as e:
        print(f"Error getting macOS proxy: {e}")

    return http_proxy, https_proxy


def get_system_proxy_details():
    """
    Attempts to retrieve HTTP and HTTPS proxy details from the system.
    Prioritizes environment variables, then OS-specific methods.
    """
    http_proxy = None
    https_proxy = None

    print("Attempting to detect proxy settings...")

    # 1. Check standard environment variables first (most common for applications)
    http_proxy = os.getenv('HTTP_PROXY') or os.getenv('http_proxy')
    https_proxy = os.getenv('HTTPS_PROXY') or os.getenv('https_proxy')

    if http_proxy or https_proxy:
        print("Found proxy details from environment variables.")
        return http_proxy, https_proxy

    # 2. OS-specific detection
    system_os = platform.system()
    if system_os == "Windows":
        print("Attempting to get proxy from Windows Registry...")
        http_proxy, https_proxy = get_windows_proxy_from_registry()
    elif system_os == "Darwin": # macOS
        print("Attempting to get proxy from macOS scutil...")
        http_proxy, https_proxy = get_macos_proxy_from_scutil()
    elif system_os == "Linux":
        print("On Linux, proxy settings are typically set via environment variables (already checked).")
        print("If a graphical desktop environment proxy is set, it might not be easily discoverable by a script.")
        # No additional reliable generic Linux proxy detection method without third-party libraries

    return http_proxy, https_proxy

def generate_batch_file(http_proxy, https_proxy):
    """
    Generates the batch file with the provided proxy details.
    """
    batch_content = [
        "@echo off",
        "",
        "pip install --upgrade certifi",
        "",
        "REM Set the trusted hosts (important for corporate networks or SSL issues)",
        "set TRUSTED_HOSTS=--trusted-host pypi.org --trusted-host files.pythonhosted.org --trusted-host wrepp0401.cpr.ca",
        ""
    ]

    if http_proxy:
        batch_content.append(f"set HTTP_PROXY={http_proxy}")
    else:
        batch_content.append("REM HTTP_PROXY not found or detected automatically.")
        batch_content.append("REM If you have an HTTP proxy, please update the line below with your proxy server details:")
        batch_content.append("set HTTP_PROXY=http://[username:password@]your.proxy.server:port  REM <-- UPDATE THIS LINE IF NEEDED")

    if https_proxy:
        batch_content.append(f"set HTTPS_PROXY={https_proxy}")
    else:
        batch_content.append("REM HTTPS_PROXY not found or detected automatically.")
        batch_content.append("REM If you have an HTTPS proxy, please update the line below with your proxy server details:")
        batch_content.append("set HTTPS_PROXY=http://[username:password@]your.proxy.server:port  REM <-- UPDATE THIS LINE IF NEEDED")

    batch_content.extend([
        "",
        "REM Install libraries",
        "pip install %TRUSTED_HOSTS% --no-warn-script-location dotenv pytest ragas sacrebleu evaluate nltk rouge_score datasets langchain-community langchain-openai",
        "",
        "REM Unset proxy variables after use (optional)",
        "set HTTP_PROXY=",
        "set HTTPS_PROXY="
    ])

    output_filename = "000_setup.bat"
    with open(output_filename, "w") as f:
        f.write("\n".join(batch_content))
    print(f"\nBatch file '{output_filename}' generated successfully!")

    if not http_proxy or not https_proxy:
        print("\nIMPORTANT: One or both proxy details were NOT automatically detected.")
        print(f"Please open '{output_filename}' and manually update the 'set HTTP_PROXY' and 'set HTTPS_PROXY' lines if needed.")
        print("Remember to include 'http://' or 'https://' prefix and potentially 'username:password@' if required.")

if __name__ == "__main__":
    detected_http_proxy, detected_https_proxy = get_system_proxy_details()
    generate_batch_file(detected_http_proxy, detected_https_proxy)
    