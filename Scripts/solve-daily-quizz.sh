#!/bin/bash

establish_session() {
	curl -k -s -c -  https://www.retoactinver.com/minisitio/reto/login/ | \
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

get_daily_quizz() {
	curl -s -k -X "GET" \
    -H "Host: www.retoactinver.com" -H "Sec-Ch-Ua: \"Not;A=Brand\";v=\"99\", \"Chromium\";v=\"106\"" -H "Accept: application/json, text/plain, */*" -H "Sec-Ch-Ua-Mobile: ?0" -H "User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/106.0.5249.62 Safari/537.36" -H "Sec-Ch-Ua-Platform: \"Windows\"" -H "Sec-Fetch-Site: same-origin" -H "Sec-Fetch-Mode: cors" -H "Sec-Fetch-Dest: empty" -H "Referer: https://www.retoactinver.com/RetoActinver/" -H "Accept-Encoding: gzip, deflate" -H "Accept-Language: es-419,es;q=0.9" -H "Connection: close" \
    -b "tokenapp=$(jq -r ".tokenApp" SessionInfo.json); TS016e21d6=$(jq -r ".TS016e21d6" SessionInfo.json); tokensesion=$(jq -r ".tokenSession" SessionInfo.json);" \
    "https://www.retoactinver.com/reto/app/quiz/consultaContestoQuizz?cveUsuario=osvaldo.hdz.m@outlook.com&cx_token_app=$(jq -r ".tokenApp" SessionInfo.json)&cx_tokenSesionApl=$(jq -r ".tokenSession" SessionInfo.json)" >  SessionInfoTmp03.json && \
    jq 'del(.collection[0].Pregunta.Pregunta.logoPatrocinador)' SessionInfoTmp03.json >  SessionInfoTmp04.json
}

send_quizz_answer() {
	IDRRESP=$(shuf -i "$(expr $(jq -r ".collection[0].Pregunta.respuestas[0].idRespuesta" SessionInfoTmp04.json) + 0)"-"$(expr $(jq -r ".collection[0].Pregunta.respuestas[0].idRespuesta" SessionInfoTmp04.json) + 2)" -n 1)

	printf "\n\nAnswering with idRespuesta: $IDRRESP"

	curl -s -k -X "POST" \
    -H "Host: www.retoactinver.com" -H "Content-Length: 0" -H "Sec-Ch-Ua: \"Not;A=Brand\";v=\"99\", \"Chromium\";v=\"106\"" -H "Accept: application/json, text/plain, */*" -H "Sec-Ch-Ua-Mobile: ?0" -H "User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/106.0.5249.62 Safari/537.36" -H "Sec-Ch-Ua-Platform: \"Windows\"" -H "Origin: https://www.retoactinver.com" -H "Sec-Fetch-Site: same-origin" -H "Sec-Fetch-Mode: cors" -H "Sec-Fetch-Dest: empty" -H "Referer: https://www.retoactinver.com/RetoActinver/" -H "Accept-Encoding: gzip, deflate" -H "Accept-Language: es-419,es;q=0.9" -H "Connection: close" \
    -b "_gcl_au=1.1.1837484389.1666105353; _ga=GA1.2.97174228.1666105354; _fbp=fb.1.1666105354438.780127227; _hjSessionUser_2963971=eyJpZCI6IjE1OTk0OTU4LWExNGEtNTVlZC04YjkwLWE5M2U3YzJkNTI5OCIsImNyZWF0ZWQiOjE2NjYxMDUzNTM0NjAsImV4aXN0aW5nIjp0cnVlfQ==; _gid=GA1.2.676068137.1666719280; _hjFirstSeen=1; _hjAbsoluteSessionInProgress=0; _hjSession_2963971=eyJpZCI6IjY5ZmU3Yzg1LWFkMjUtNDIzNC1iODU4LTBkYzkwNDEwZTIzYyIsImNyZWF0ZWQiOjE2NjY3MTkzMTkyNjQsImluU2FtcGxlIjpmYWxzZX0=; cxCveUsuario=osvaldo.hdz.m%40outlook.com; nickName=osvaldohm23; nombre=Osvaldo; apellidoPaterno=Hern%C3%A1ndez; apellidoMaterno=Morales; idUsuario=341478; nivel=principiante; iva=16; comision=0.1; idGrupo=null; grupo=null; idPosicion=undefined; idPerfil=9; esGratis=NO; tipoPersona=RA; _hjSessionUser_3213375=eyJpZCI6IjVlN2M3MmUxLThiNDQtNTgwMC05ZTczLTBkOGExZWJhYmNmZCIsImNyZWF0ZWQiOjE2NjY3MTkyNzk5NDksImV4aXN0aW5nIjp0cnVlfQ==; _hjSession_3213375=eyJpZCI6ImVmNjZmNzY2LTNlNWYtNDRlMi1iNTZjLTFiNDA2N2RkZDA0NSIsImNyZWF0ZWQiOjE2NjY3MjMzNzAzNTcsImluU2FtcGxlIjpmYWxzZX0=; ADRUM=s=1666724841239&r=https%3A%2F%2Fwww.retoactinver.com%2FRetoActinver%2F%3Fhash%3D-1552272943; _hjIncludedInSessionSample=0; _gat_UA-150830483-8=1; tokenapp=$(jq -r ".tokenApp" SessionInfo.json); _gat_gtag_UA_150830483_9=1; TS016e21d6=$(jq -r ".TS016e21d6" SessionInfo.json); tokensesion=$(jq -r ".tokenSession" SessionInfo.json)" \
    "https://www.retoactinver.com/reto/app/quiz/contestarQuiz?cveUsuario=osvaldo.hdz.m@outlook.com&cx_tokenSesionApl=$(jq -r ".tokenSession" SessionInfo.json)&cx_token_app=$(jq -r ".tokenApp" SessionInfo.json)&idRespuesta=$IDRRESP&tokenApp=$(jq -r ".tokenApp" SessionInfo.json)&tokenSession=$(jq -r ".tokenSession" SessionInfo.json)"
}

check_file() {
if [ ! -f $1 ]; then
    printf "\nFile not found!"
    exit 2
fi
}

help() {
   # Display Help
   printf "\nSyntax: SolveDailyQuizz.sh [-h]"
   printf "\nOptions:"
   printf "\n-h	Print this Help."
}

SHORT=h
LONG=help
OPTS=$(getopt -a -n SolveDailyQuizz.sh --options $SHORT --longoptions $LONG -- "$@")

eval set -- "$OPTS"

printf "\n\n------------------------ Solve Daily Quizz ---------------------"

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
cat SessionInfo.json
printf "\n\n[$(date +'%r')] Getting daily quizz...\n"
get_daily_quizz
cat SessionInfoTmp04.json
#printf "\n%s " "Press enter to continue"
#read ans
printf "\n\n[$(date +'%r')] Recovering session for next operation and updating session SessionInfo.json...\n"
recover_session
cat SessionInfo.json
printf "\n\n[$(date +'%r')] Answering quizz.json\n"
send_quizz_answer
printf "\n\n[$(date +'%r')] Closing session...\n"
close_session
printf "\n\n[$(date +'%r')] Deleting temporal files...\n"
rm SessionInfo*
