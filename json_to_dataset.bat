@echo off
set a=0
set str1=.json
set str2=haha
:loop
set /a a+=1
echo.%a%
set "str2=%a%%str1%"
labelme_json_to_dataset %str2%
if %a% == 41 goto end
goto loop
