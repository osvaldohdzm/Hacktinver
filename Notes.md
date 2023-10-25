
## Estraegias SWING TRADING 

en la bolsa 1 acciones 

configuracion estado estado tecnico 
- venta
- venta 
- compra

indices etf especufixos 

- SOXL
- TEC 


GDX
SLV
TLT

SOXL SOXS
TECL TECS
SPXL SPXS TQQQ


## URL

https://www.retoactinver.com/minisitio/reto/login.html


1. We need to provide a new *TS016e21d6* cookie to works, this could be one option.
 
 ```
 curl -s -c -  https://www.retoactinver.com/minisitio/reto/login.html | grep -E 'TS016e21d6' | sed "s/.*TS016e21d6\t//g"
 ```
 
 The we can obtain our session token:

2. With this comand curl:

```
curl -i -s -k -X $'POST' \
    -H $'Host: www.retoactinver.com' -H $'Content-Length: 64' -H $'Sec-Ch-Ua: \"Not;A=Brand\";v=\"99\", \"Chromium\";v=\"106\"' -H $'Accept: application/json, text/javascript, */*; q=0.01' -H $'Content-Type: application/json' -H $'X-Requested-With: XMLHttpRequest' -H $'Sec-Ch-Ua-Mobile: ?0' -H $'User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/106.0.5249.62 Safari/537.36' -H $'Sec-Ch-Ua-Platform: \"Windows\"' -H $'Origin: https://www.retoactinver.com' -H $'Sec-Fetch-Site: same-origin' -H $'Sec-Fetch-Mode: cors' -H $'Sec-Fetch-Dest: empty' -H $'Referer: https://www.retoactinver.com/minisitio/reto/login.html' -H $'Accept-Encoding: gzip, deflate' -H $'Accept-Language: es-419,es;q=0.9' -H $'Connection: close' \
    -b $'TS016e21d6=0121f724fc807ac7dc772edfedfb315f88db10b75dbad67c6496e14f8cd8c29efe89e9818040c9628f6ac8cb55b53c905e5691636f' \
    --data-binary $'{\"usuario\":\"exampleuser@mail\",\"password\":\"examplespassword\"}' \
    $'https://www.retoactinver.com/reto/app/usuarios/login'
 ```

### Start Session  make session file (with proxy)

```
curl --proxy "http://172.17.208.1:8081" -k -s -c -  https://www.retoactinver.com/minisitio/reto/login.html | grep -E 'TS016e21d6' | sed "s/.*TS016e21d6\t//g" | xargs -I %arg%  echo "{ \"TS016e21d6\" : \"%arg%\" }" > SessionInfoTmp01.json && curl --proxy "http://172.17.208.1:8081" -k -s -X $'POST' -H $'Host: www.retoactinver.com' -H $'Content-Length: 64' -H $'Sec-Ch-Ua: \'Not;A=Brand\';v=\'99\', \'Chromium\';v=\'106\'' -H $'Accept: application/json, text/javascript, */*; q=0.01' -H $'Content-Type: application/json' -H $'X-Requested-With: XMLHttpRequest' -H $'Sec-Ch-Ua-Mobile: ?0' -H $'User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/106.0.5249.62 Safari/537.36' -H $'Sec-Ch-Ua-Platform: \'Windows\'' -H $'Origin: https://www.retoactinver.com' -H $'Sec-Fetch-Site: same-origin' -H $'Sec-Fetch-Mode: cors' -H $'Sec-Fetch-Dest: empty' -H $'Referer: https://www.retoactinver.com/minisitio/reto/login.html' -H $'Accept-Encoding: gzip, deflate' -H $'Accept-Language: es-419,es;q=0.9' -H $'Connection: close' -b $'TS016e21d6=$(jq ".TS016e21d6" SessionInfoTmp01.json)' --data-binary $'{\"usuario\":\"osvaldo.hdz.m@outlook.com\",\"password\":\"Os23valdo1.\"}'  $'https://www.retoactinver.com/reto/app/usuarios/login' > SessionInfoTmp02.json && jq -s '.[0] * .[1]' SessionInfoTmp01.json  SessionInfoTmp02.json > SessionInfo.json && rm SessionInfoTmp*
```

### Close Session (with proxy)

```
curl --proxy "http://172.17.208.1:8081" -k -i -s -X $'POST' \
    -H $'Host: www.retoactinver.com' -H $'Content-Length: 0' -H $'Sec-Ch-Ua: \"Not;A=Brand\";v=\"99\", \"Chromium\";v=\"106\"' -H $'Accept: application/json, text/plain, */*' -H $'Adrum: isAjax:true' -H $'Sec-Ch-Ua-Mobile: ?0' -H $'User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/106.0.5249.62 Safari/537.36' -H $'Sec-Ch-Ua-Platform: \"Windows\"' -H $'Origin: https://www.retoactinver.com' -H $'Sec-Fetch-Site: same-origin' -H $'Sec-Fetch-Mode: cors' -H $'Sec-Fetch-Dest: empty' -H $'Referer: https://www.retoactinver.com/RetoActinver/' -H $'Accept-Encoding: gzip, deflate' -H $'Accept-Language: es-419,es;q=0.9' -H $'Connection: close' \
    -b $"TS016e21d6=$(jq -r ".TS016e21d6" SessionInfo.json);  tokenapp=$(jq -r ".tokenApp" SessionInfo.json); tokensesion=$(jq -r ".tokenSession" SessionInfo.json)" \
    $"https://www.retoactinver.com/reto/app/usuarios/session/closeSesion?user=osvaldo.hdz.m@outlook.com&tokenSession=1666120956790&tokenApp=$(jq -r ".tokenApp" SessionInfo.json)"

```

-----

### Get token for new operation (PROXY)
```  
jq '.tokenApp' SessionInfo.json | xargs -I %arg% curl --proxy "http://172.17.208.1:8081" -k -X $'POST' -H $'Host: www.retoactinver.com' -H $'Content-Length: 0' -H $'Sec-Ch-Ua: \"Not;A=Brand\";v=\"99\", \"Chromium\";v=\"106\"' -H $'Accept: application/json, text/plain, */*' -H $'Sec-Ch-Ua-Mobile: ?0' -H $'User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/106.0.5249.62 Safari/537.36' -H $'Sec-Ch-Ua-Platform: \"Windows\"' -H $'Origin: https://www.retoactinver.com' -H $'Sec-Fetch-Site: same-origin' -H $'Sec-Fetch-Mode: cors' -H $'Sec-Fetch-Dest: empty' -H $'Referer: https://www.retoactinver.com/RetoActinver/' -H $'Accept-Encoding: gzip, deflate' -H $'Accept-Language: es-419,es;q=0.9' -H $'Connection: close' -b "TS016e21d6=$(jq ".TS016e21d6" SessionInfo.json);tokenapp=$(jq ".tokenApp" SessionInfo.json); tokensesion=$(jq ".tokenSession" SessionInfo.json); "  "https://www.retoactinver.com/reto/app/usuarios/session/recoveryTokenSession?user=osvaldo.hdz.m@outlook.com&tokenApp=%arg%" > SessionInfoTmp02.json && jq '.tokenSession = "$(jq ".cxValue" SessionInfoTmp02.json)"' SessionInfo.json
```



```
curl -i -s -k -X $'POST'     -H $'Host: www.retoactinver.com' -H $'Content-Length: 0' -H $'Sec-Ch-Ua: \"Not;A=Brand\";v=\"99\", \"Chromium\";v=\"106\"' -H $'Accept: application/json, text/plain, */*' -H $'Adrum: isAjax:true' -H $'Sec-Ch-Ua-Mobile: ?0' -H $'User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/106.0.5249.62 Safari/537.36' -H $'Sec-Ch-Ua-Platform: \"Windows\"' -H $'Origin: https://www.retoactinver.com' -H $'Sec-Fetch-Site: same-origin' -H $'Sec-Fetch-Mode: cors' -H $'Sec-Fetch-Dest: empty' -H $'Referer: https://www.retoactinver.com/RetoActinver/' -H $'Accept-Encoding: gzip, deflate' -H $'Accept-Language: es-419,es;q=0.9' -H $'Connection: close'     -b $"TS016e21d6=0121f724fc48d172308c42debeb75399001e14665cd4d4cc2071a7c800d07d7573296bbcaa21f8da737045798421c6a7a3a7a7b72d;tokenapp=B7E87F5B53AB12BAA572F8D83B3E4590; tokensesion=$(jq -r ".tokenSession" SessionInfo.json)"     $"https://www.retoactinver.com/reto/app/quiz/contestarQuiz?cveUsuario=osvaldo.hdz.m@outlook.com&cx_tokenSesionApl=$(jq -r ".tokenSession" SessionInfo.json)&cx_token_app=B7E87F5B53AB12BAA572F8D83B3E4590&idRespuesta=324&tokenApp=B7E87F5B53AB12BAA572F8D83B3E4590&tokenSession=$(jq -r ".tokenSession" SessionInfo.json)"
```



Lo que vari es el tokenSession
