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

### Start Session without Proxy make session file
```
curl -k -s -c -  https://www.retoactinver.com/minisitio/reto/login.html | grep -E 'TS016e21d6' | sed "s/.*TS016e21d6\t//g" | xargs -I %arg%  echo "{ \"TS016e21d6\" : \"%arg%\" }" > SessionInfoTmp01.json && curl -k -s -X $'POST' -H $'Host: www.retoactinver.com' -H $'Content-Length: 64' -H $'Sec-Ch-Ua: \'Not;A=Brand\';v=\'99\', \'Chromium\';v=\'106\'' -H $'Accept: application/json, text/javascript, */*; q=0.01' -H $'Content-Type: application/json' -H $'X-Requested-With: XMLHttpRequest' -H $'Sec-Ch-Ua-Mobile: ?0' -H $'User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/106.0.5249.62 Safari/537.36' -H $'Sec-Ch-Ua-Platform: \'Windows\'' -H $'Origin: https://www.retoactinver.com' -H $'Sec-Fetch-Site: same-origin' -H $'Sec-Fetch-Mode: cors' -H $'Sec-Fetch-Dest: empty' -H $'Referer: https://www.retoactinver.com/minisitio/reto/login.html' -H $'Accept-Encoding: gzip, deflate' -H $'Accept-Language: es-419,es;q=0.9' -H $'Connection: close'     -b $'TS016e21d6=$(jq ".TS016e21d6" SessionInfoTmp01.json)' --data-binary $'{\"usuario\":\"osvaldo.hdz.m@outlook.com\",\"password\":\"Os23valdo1.\"}'  $'https://www.retoactinver.com/reto/app/usuarios/login' >> SessionInfoTmp02.json && jq -s '.[0] * .[1]' SessionInfoTmp01.json  SessionInfoTmp02.json > SessionInfo.json && rm SessionInfoTmp*
```
