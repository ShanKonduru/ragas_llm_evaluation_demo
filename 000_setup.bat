@echo off

pip install --upgrade certifi

REM Set the trusted hosts (important for corporate networks or SSL issues)
set TRUSTED_HOSTS=--trusted-host pypi.org --trusted-host files.pythonhosted.org --trusted-host wrepp0401.cpr.ca

REM Install openpyxl
pip install %TRUSTED_HOSTS% --no-warn-script-location dotenv pytest ragas sacrebleu evaluate nltk rouge_score datasets langchain-community langchain-openai

@echo off

pip install --upgrade certifi

REM Set the trusted hosts (important for corporate networks or SSL issues)
set TRUSTED_HOSTS=--trusted-host pypi.org --trusted-host files.pythonhosted.org --trusted-host wrepp0401.cpr.ca

REM Set proxy variables
set HTTP_PROXY=http://[username:password@]your.proxy.server:port
set HTTPS_PROXY=http://[username:password@]your.proxy.server:port

REM Install libraries
pip install %TRUSTED_HOSTS% --no-warn-script-location dotenv pytest ragas sacrebleu evaluate nltk rouge_score datasets langchain-community langchain-openai

REM Unset proxy variables after use (optional)
set HTTP_PROXY=
set HTTPS_PROXY=
