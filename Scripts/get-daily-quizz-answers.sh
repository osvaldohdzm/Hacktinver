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

get_quizz_answers() {
	curl -s "https://www.retoactinver.com/reto/app/quiz/consultaMisRespuestas?cveUsuario=osvaldo.hdz.m@outlook.com&cx_token_app=$(jq -r ".tokenApp" SessionInfo.json)&cx_tokenSesionApl=$(jq -r ".tokenSession" SessionInfo.json)" \
  -H "authority: www.retoactinver.com" \
  -H "accept: application/json, text/plain, */*" \
  -H "accept-language: es-419,es;q=0.9" \
  -H "cookie: tokenapp=$(jq -r ".tokenApp" SessionInfo.json); TS016e21d6=$(jq -r ".TS016e21d6" SessionInfo.json)" \                                   -H "referer: https://www.retoactinver.com/RetoActinver/" \
  -H "sec-ch-ua: \"Not;A=Brand\";v=\"99\", \"Chromium\";v=\"106\"" \
  -H "sec-ch-ua-mobile: ?0" \
  -H "sec-ch-ua-platform: "Windows"" \
  -H "sec-fetch-dest: empty" \
  -H "sec-fetch-mode: cors" \
  -H "sec-fetch-site: same-origin" \
  -H "user-agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/106.0.5249.62 Safari/537.36" \
  --compressed \
  --insecure | \
  jq
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
printf "\n\n[$(date +'%r')] Getting quizz answers...\n"
get_quizz_answers
printf "\n\n[$(date +'%r')] Recovering session for next operation and updating session SessionInfo.json...\n"
recover_session
cat SessionInfo.json
printf "\n\n[$(date +'%r')] Closing session...\n"
close_session
printf "\n\n[$(date +'%r')] Deleting temporal files...\n"
rm SessionInfo*
