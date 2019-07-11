@echo off
set a=0
set str1=C:\Users\12084\Desktop\Proj\data\7-11\
set str2=dataset\
set str3=mask\
set str4=_json\
set str5=label.png
set str6=.png
:loop
set /a a+=1
set str10="%q%%str1%%str2%%a%%str4%%str5%"
set str11="%q%%str1%%str3%%a%%str6%"
copy %str10% %str11%
if %a% == 41 goto end
goto loop
:end
exit