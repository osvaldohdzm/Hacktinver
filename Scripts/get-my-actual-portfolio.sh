#/usr/bin/bash

establish_session() {
	curl -k -s -c -  https://www.retoactinver.com/minisitio/reto/login.html | \
	grep -E 'TS016e21d6' | \
	sed "s/.*TS016e21d6\t//g" | \
	xargs -I %arg%  printf "\n{ \"TS016e21d6\" : \"%arg%\" }" > SessionInfoTmp01.json && \
	curl -s -X 'POST' -H 'Host: www.retoactinver.com' -H 'Content-Length: 64' -H 'Sec-Ch-Ua: Not;A=Brand;v=99, Chromium;v=106' -H 'Accept: application/json, text/javascript, */*; q=0.01' -H 'Content-Type: application/json' -H 'X-Requested-With: XMLHttpRequest' -H 'Sec-Ch-Ua-Mobile: ?0' -H 'User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/106.0.5249.62 Safari/537.36' -H 'Sec-Ch-Ua-Platform: Windows' -H 'Origin: https://www.retoactinver.com' -H 'Sec-Fetch-Site: same-origin' -H 'Sec-Fetch-Mode: cors' -H 'Sec-Fetch-Dest: empty' -H 'Referer: https://www.retoactinver.com/minisitio/reto/login.html' -H 'Accept-Encoding: gzip, deflate' -H 'Accept-Language: es-419,es;q=0.9' -H 'Connection: close' -b "TS016e21d6=$(jq ".TS016e21d6" SessionInfoTmp01.json)" --data-binary "{\"usuario\":\"osvaldo.hdz.m@outlook.com\",\"password\":\"Os23valdo1.\"}" 'https://www.retoactinver.com/reto/app/usuarios/login' > SessionInfoTmp02.json && \
	jq -s '.[0] * .[1]' SessionInfoTmp01.json  SessionInfoTmp02.json > SessionInfo.json 
}

recover_session() {
	curl -s -k -X "POST" \
    -H "Host: www.retoactinver.com" -H "Content-Length: 0" -H "Sec-Ch-Ua: \"Not;A=Brand\";v=\"99\", \"Chromium\";v=\"106\"" -H "Accept: application/json, text/plain, */*" -H "Sec-Ch-Ua-Mobile: ?0" -H "User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/106.0.5249.62 Safari/537.36" -H "Sec-Ch-Ua-Platform: \"Windows\"" -H "Origin: https://www.retoactinver.com" -H "Sec-Fetch-Site: same-origin" -H "Sec-Fetch-Mode: cors" -H "Sec-Fetch-Dest: empty" -H "Referer: https://www.retoactinver.com/RetoActinver/" -H "Accept-Encoding: gzip, deflate" -H "Accept-Language: es-419,es;q=0.9" -H "Connection: close" \
    -b "tokenapp=$(jq -r ".tokenApp" SessionInfo.json); TS016e21d6=$(jq -r ".TS016e21d6" SessionInfo.json); tokensesion=$(jq -r ".tokenSession" SessionInfo.json)" \
    "https://www.retoactinver.com/reto/app/usuarios/session/recoveryTokenSession?user=osvaldo.hdz.m@outlook.com&tokenApp=$(jq -r ".tokenApp" SessionInfo.json)" > SessionInfoTmp02.json && \
    jq ".tokenSession = "$(jq ".cxValue" SessionInfoTmp02.json)"" SessionInfo.json > SessionInfoTmp05.json && \
    mv SessionInfoTmp05.json SessionInfo.json
}

close_session() {
	curl -s -k -X "POST"  \
		-H "Host: www.retoactinver.com" -H "Content-Length: 0" -H "Sec-Ch-Ua: \"Not;A=Brand\";v=\"99\", \"Chromium\";v=\"106\"" -H "Accept: application/json, text/plain, */*" -H "Sec-Ch-Ua-Mobile: ?0" -H "User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/106.0.5249.62 Safari/537.36" -H "Sec-Ch-Ua-Platform: \"Windows\"" -H "Origin: https://www.retoactinver.com" -H "Sec-Fetch-Site: same-origin" -H "Sec-Fetch-Mode: cors" -H "Sec-Fetch-Dest: empty" -H "Referer: https://www.retoactinver.com/RetoActinver/" -H "Accept-Encoding: gzip, deflate" -H "Accept-Language: es-419,es;q=0.9" -H "Connection: close" \
		-b "TS016e21d6=$(jq -r ".TS016e21d6" SessionInfo.json); tokensesion=$(jq -r ".tokenSession" SessionInfo.json)" \
		"https://www.retoactinver.com/reto/app/usuarios/session/closeSesion?user=osvaldo.hdz.m@outlook.com&tokenSession=$(jq -r ".tokenSession" SessionInfo.json)&tokenApp=$(jq -r ".tokenApp" SessionInfo.json)"
}


get_portfolio(){
curl -s "https://www.retoactinver.com/reto/app/usuarios/compra/portafolios?usuario=osvaldo.hdz.m@outlook.com&tokenApp=$(jq -r ".tokenApp" SessionInfo.json)&tokenSession=$(jq -r ".tokenSession" SessionInfo.json)" \
  -X "POST" \
  -H "authority: www.retoactinver.com" \
  -H "accept: application/json, text/plain, */*" \
  -H "accept-language: es-419,es;q=0.9" \
  -H "adrum: isAjax:true" \
  -H "content-length: 0" \
  -H "cookie: _gcl_au=1.1.1837484389.1666105353; _ga=GA1.2.97174228.1666105354; _fbp=fb.1.1666105354438.780127227; _hjSessionUser_2963971=eyJpZCI6IjE1OTk0OTU4LWExNGEtNTVlZC04YjkwLWE5M2U3YzJkNTI5OCIsImNyZWF0ZWQiOjE2NjYxMDUzNTM0NjAsImV4aXN0aW5nIjp0cnVlfQ==; _hjSessionUser_3213375=eyJpZCI6IjVlN2M3MmUxLThiNDQtNTgwMC05ZTczLTBkOGExZWJhYmNmZCIsImNyZWF0ZWQiOjE2NjY3MTkyNzk5NDksImV4aXN0aW5nIjp0cnVlfQ==; _gid=GA1.2.984536386.1667937964; cxCveUsuario=osvaldo.hdz.m%40outlook.com; nickName=osvaldohm23; nombre=Osvaldo; apellidoPaterno=Hern%C3%A1ndez; apellidoMaterno=Morales; idUsuario=341478; nivel=principiante; iva=16; comision=0.1; idGrupo=null; grupo=null; idPosicion=undefined; idPerfil=9; esGratis=NO; tipoPersona=RA; _hjSession_3213375=eyJpZCI6IjY2ZTNmNGNkLTE2YjktNGQwNy04OGYzLWU5ZWE1YmJjMzZhOCIsImNyZWF0ZWQiOjE2Njc5NDYxOTA3MTgsImluU2FtcGxlIjpmYWxzZX0=; _hjAbsoluteSessionInProgress=1; _hjSession_2963971=eyJpZCI6IjcxODc5ZGVlLWM3NzItNGNjOS1hM2Y2LTJlZjM2NTA4YzA3MCIsImNyZWF0ZWQiOjE2Njc5NDYxOTM1MjksImluU2FtcGxlIjpmYWxzZX0=; ADRUM=s=1667946794182&r=https%3A%2F%2Fwww.retoactinver.com%2FRetoActinver%2F%3Fhash%3D264197004; _hjIncludedInPageviewSample=1; _hjIncludedInSessionSample=0; TS016e21d6=$(jq -r ".TS016e21d6" SessionInfo.json); tokenapp=$(jq -r ".tokenApp" SessionInfo.json); _gat_gtag_UA_150830483_9=1; tokensesion=$(jq -r ".tokenSession" SessionInfo.json)" \
  -H "origin: https://www.retoactinver.com" \
  -H "referer: https://www.retoactinver.com/RetoActinver/" \
  -H "sec-ch-ua: \"Chromium\";v=\"107\", \"Not=A?Brand\";v=\"24\"" \
  -H "sec-ch-ua-mobile: ?0" \
  -H "sec-ch-ua-platform: \"Windows\"" \
  -H "sec-fetch-dest: empty" \
  -H "sec-fetch-mode: cors" \
  -H "sec-fetch-site: same-origin" \
  -H "user-agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.5304.63 Safari/537.36" \
  --compressed \
  --insecure | \
    jq '.collection' > portfolio.json
}

check_file() {
if [ ! -f $1 ]; then
    printf "\nFile not found!"
    exit 2
fi
}

help() {
   # Display Help
   printf
   printf "\nSyntax: SolveDailyQuizz.sh [-h]"
   printf "\nOptions:"
   printf "\n-h	Print this Help."
   printf
}

SHORT=h
LONG=help
OPTS=$(getopt -a -n SolveDailyQuizz.sh --options $SHORT --longoptions $LONG -- "$@")

eval set -- "$OPTS"

printf "\n\n------------------------ Get Daily Quizz ---------------------"

# Get the options
while getopts ":h" option; do
   case $option in
      h) # display Help
         help
         exit;;
   esac
done

printf "\n\n[$(date +'%r')] Getting login parameters in SessionInfo.json...\n"
establish_session
cat SessionInfo.json
printf "\n\n[$(date +'%r')] Recovering session for next operation and updating session SessionInfo.json...\n"
recover_session
printf "\n\n[$(date +'%r')] Getting portfolio...\n"
get_portfolio
cat portfolio.json
printf "\n\n[$(date +'%r')] Closing session...\n"
close_session
printf "\n\n[$(date +'%r')] Deleting temporal files...\n"
rm SessionInfo*
