@echo off
rem run_all.bat - Executes a list of batch files sequentially in separate command windows.

rem Define the list of batch files to execute.
rem IMPORTANT: Replace these with the actual paths/names of your batch files.
rem If the batch files are not in the same directory as run_all.bat,
rem you must provide their full paths (e.g., C:\path\to\your\batch_file_1.bat)
set "LIST_OF_BATS_TO_RUN=1_demo.bat 2_demo.bat 3_demo.bat 4_demo.bat 5_demo.bat 6_demo.bat 7_demo.bat"

echo Starting sequential execution of batch files...
echo.

rem Loop through the defined batch files and execute each one
rem '%%f' will represent each file name in the LIST_OF_BATS_TO_RUN variable.
for %%f in (%LIST_OF_BATS_TO_RUN%) do (
    echo Running: "%%f"
    rem The 'start' command opens a new window.
    rem '/wait' ensures run_all.bat waits for the new window process to close.
    rem 'cmd /k' keeps the new command window open after the batch file finishes.
    rem The title of the new window will be the batch file name.
    start "Running: %%~nf" cmd /k call "%%f"
    echo.
)

echo All specified batch files have completed.
pause
