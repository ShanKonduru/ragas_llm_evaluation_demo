@echo off

pip install --upgrade certifi

REM Set the trusted hosts (important for corporate networks or SSL issues)
set TRUSTED_HOSTS=--trusted-host pypi.org --trusted-host files.pythonhosted.org --trusted-host wrepp0401.cpr.ca

REM Install openpyxl
pip install %TRUSTED_HOSTS% --no-warn-script-location dotenv pytest ragas sacrebleu evaluate nltk rouge_score datasets langchain-community langchain-openai

REM @echo off
REM pip install --upgrade certifi
REM REM Set the trusted hosts (important for corporate networks or SSL issues)
REM set TRUSTED_HOSTS=--trusted-host pypi.org --trusted-host files.pythonhosted.org --trusted-host wrepp0401.cpr.ca

REM REM Set proxy variables
REM set HTTP_PROXY=http://[username:password@]your.proxy.server:port
REM set HTTPS_PROXY=http://[username:password@]your.proxy.server:port

REM REM Install libraries
REM pip install %TRUSTED_HOSTS% --no-warn-script-location dotenv pytest ragas sacrebleu evaluate nltk rouge_score datasets langchain-community langchain-openai

REM REM Unset proxy variables after use (optional)
REM set HTTP_PROXY=
REM set HTTPS_PROXY=
