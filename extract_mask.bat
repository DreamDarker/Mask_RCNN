@echo off
set a=300
set str1=C:\Users\12084\Desktop\Proj\data\9-8\
set str2=dataset\
set str3=mask\
set str4=_json\
set str7=img\
set str8=img.png
set str5=label.png
set str6=.png
:loop
set /a a+=1
set str10="%q%%str1%%str2%%a%%str4%%str5%"
set str11="%q%%str1%%str3%%a%%str6%"
set str12="%q%%str1%%str2%%a%%str4%%str8%"
set str13="%q%%str1%%str7%%a%%str6%"
copy %str10% %str11%
copy %str12% %str13%
echo %a%
if %a% == 1800 goto end
goto loop
:end
exit