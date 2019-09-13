@echo off
set a=0
set b=300
set c=0
set root=C:\Users\12084\Desktop\Proj\data\9-8\
set dataset=dataset\
set json=_json\
set yaml=info.yaml
:loop1
set /a a+=1
:loop2
set /a b+=1
echo %b%
set str00="%q%%root%%dataset%%b%%json%"
md %str00%
set str14="%q%%root%%dataset%%a%%json%%yaml%"
set str15="%q%%root%%dataset%%b%%json%%yaml%"
copy %str14% %str15%
if %b% == 310 goto end
set /a c = ((%b%-300)/%a%)-4
if %c% == 1  goto loop1
goto loop2
:end