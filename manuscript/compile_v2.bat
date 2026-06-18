@echo off
cd /d "%~dp0"
set BIN=..\.tools\tinytex\dist\TinyTeX\bin\windows
%BIN%\pdflatex.exe -interaction=nonstopmode -jobname=main_v2 main.tex > nul 2>&1
%BIN%\bibtex.exe main_v2 > nul 2>&1
%BIN%\pdflatex.exe -interaction=nonstopmode -jobname=main_v2 main.tex > nul 2>&1
%BIN%\pdflatex.exe -interaction=nonstopmode -jobname=main_v2 main.tex 2>&1 | findstr /c:"Output" /c:"Fatal"
