package main

import (
	"flag"
	"fmt"
	"io/ioutil"
	"net/http"
	"os"
	"strconv"
	"time"
    "os/exec"
    "strings"
	"bytes"
)

var (
	helpFlag = flag.Bool("h", false, "Show help message")
	testFlag = flag.Bool("test", false, "Perform a test GET request")
)

type SessionInfo struct {
	Tokensesion string
	TokenApp    string
}

func main() {
	flag.Parse()

	if *helpFlag {
		showHelp()
		return
	}

	if *testFlag {
		doTestRequest()
		return
	}

	fmt.Println("\n------------------------ Solve Daily Quizz ---------------------")

	menu()
}

func menu() {
	for {
		fmt.Println("\nSelecciona una opción")
		fmt.Println("1 - Iniciar sesión en la plataforma del reto")
		fmt.Println("2 - Mostrar sugerencias de compra")
		fmt.Println("3 - Mostrar portafolio actual")
		fmt.Println("4 - Comprar acciones")
		fmt.Println("5 - Mostrar órdenes")
		fmt.Println("6 - Monitorear venta")
		fmt.Println("7 - Vender todas las posiciones en el portafolio (a precio del mercado)")
		fmt.Println("8 - Restaurar sesión en plataforma del reto")
		fmt.Println("9 - Optimizar portafolio")
		fmt.Println("10 - Prueba de conexión (Reto actinver 2023)")
		fmt.Println("11 - Resolver quizz diario (Reto actinver 2023)")
		fmt.Println("12 - Prueba de Inicio de sesión (Reto actinver 2023)")
		fmt.Println("13 - Prueba de Cierre de sesión (Reto actinver 2023)")
		fmt.Println("0 - Salir")
		fmt.Print("Selecciona una opción >> ")

		var option int
		_, err := fmt.Scanf("%d", &option)
		if err != nil {
			fmt.Println("Error:", err)
			continue
		}

		switch option {
		case 1:
			fmt.Println("Iniciando sesión...")
			// Lógica para iniciar sesión
		case 2:
			fmt.Println("Mostrando sugerencias de compra...")
			// Lógica para mostrar sugerencias
		case 3:
			fmt.Println("Mostrando portafolio actual...")
			// Lógica para mostrar el portafolio
		case 4:
			fmt.Println("Comprando acciones...")
			// Lógica para comprar acciones
		case 5:
			fmt.Println("Mostrando órdenes...")
			// Lógica para mostrar órdenes
		case 6:
			fmt.Println("Monitoreando venta...")
			// Lógica para monitorear venta
		case 7:
			fmt.Println("Vendiendo todas las posiciones en el portafolio (a precio del mercado)...")
			// Lógica para vender posiciones
		case 8:
			fmt.Println("Restaurando sesión en plataforma del reto...")
			// Lógica para restaurar sesión
		case 9:
			fmt.Println("Optimizando portafolio...")
		case 10:
			fmt.Println("Realizando prueba de conexión...")
			doTestRequest()	
		case 11:
			fmt.Printf("\n\n[%s] Getting login parameters, creating LoginInfo.json...\n", time.Now().Format("03:04:05 PM"))
			getLoginInfo()
			establishSession()
			fmt.Println("Resolviendo quizz diario...")

			// Obtiene los parámetros de inicio de sesión
			sessionInfo := &SessionInfo{
				Tokensesion: "1697044025591",
				TokenApp:    "524E8A8C6CB424502EDFFE008B5936FA",
			}

			// Envía la respuesta al cuestionario
			resp, err := sendQuizzAnswer(sessionInfo, 398)
			if err != nil {
				fmt.Println(err)
				os.Exit(1)
			}

			// Verifica el código de estado de la respuesta
			if resp.StatusCode != http.StatusOK {
				fmt.Println("Error al enviar la respuesta al cuestionario: ", resp.StatusCode)
				os.Exit(1)
			}

			fmt.Println("¡La respuesta al cuestionario ha sido enviada!")		
		case 12: 	
		fmt.Printf("\n\n[%s] Getting login parameters, creating LoginInfo.json...\n", time.Now().Format("03:04:05 PM"))			
			getLoginInfo()	
		case 13: 				
		var tokensesion, tokenApp string

		fmt.Println("Enter the value for tokensesion:")
		fmt.Scan(&tokensesion)
	
		fmt.Println("Enter the value for tokenApp:")
		fmt.Scan(&tokenApp)
	
		// Ahora que tienes los valores ingresados por el usuario, puedes llamar a la función closeActinverContestSession
		fmt.Printf("\n[%s] Closing session...\n", time.Now().Format("03:04:05 PM"))
		closeActinverContestSession(tokensesion, tokenApp)
		case 0:
			fmt.Println("Saliendo del programa.")
			return
		default:
			fmt.Println("Opción no válida. Inténtalo de nuevo.")
		}
	}
}


func closeActinverContestSession(tokensesion, tokenApp string) {
    url := fmt.Sprintf("https://www.retoactinver.com/reto/app/usuarios/session/closeSesion?user=osvaldo.hdz.m@outlook.com&tokenSession=%s&tokenApp=%s", tokensesion, tokenApp)

    client := &http.Client{}
    req, err := http.NewRequest("POST", url, nil)
    if err != nil {
        fmt.Println("Error al crear la solicitud HTTP:", err)
        return
    }

    // Configurar las cabeceras de la solicitud
    req.Header.Set("Host", "www.retoactinver.com")
    req.Header.Set("Content-Length", "0")
    req.Header.Set("Accept", "application/json, text/plain, */*")
    req.Header.Set("Sec-Ch-Ua-Mobile", "?0")
    req.Header.Set("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.5845.111 Safari/537.36")
    req.Header.Set("Origin", "https://www.retoactinver.com")
    req.Header.Set("Sec-Fetch-Site", "same-origin")
    req.Header.Set("Sec-Fetch-Mode", "cors")
    req.Header.Set("Sec-Fetch-Dest", "empty")
    req.Header.Set("Referer", "https://www.retoactinver.com/RetoActinver/")
    req.Header.Set("Accept-Encoding", "gzip, deflate")
    req.Header.Set("Accept-Language", "en-US,en;q=0.9")
    req.Header.Set("Connection", "close")

    // Configurar las cookies
    cookies := fmt.Sprintf("f5avraaaaaaaaaaaaaaaa_session_=%s; tokensesion=%s", tokensesion, tokensesion)
    req.Header.Set("Cookie", cookies)

    resp, err := client.Do(req)
    if err != nil {
        fmt.Println("Error al realizar la solicitud HTTP:", err)
        return
    }
    defer resp.Body.Close()

    body, err := ioutil.ReadAll(resp.Body)
    if err != nil {
        fmt.Println("Error al leer la respuesta HTTP:", err)
        return
    }

    fmt.Println("\nRespuesta del servidor:")
    fmt.Println(string(body))

    fmt.Println("\nSesión cerrada exitosamente")
}

func sendQuizzAnswer(sessionInfo *SessionInfo, idRespuesta int) (*http.Response, error) {
	// Crea una nueva solicitud HTTP
	req, err := http.NewRequest("POST", "https://www.retoactinver.com/reto/app/quiz/contestarQuiz", nil)
	if err != nil {
		return nil, err
	}

	// Agrega los encabezados necesarios
	req.Header.Add("Host", "www.retoactinver.com")
	req.Header.Add("Content-Length", "0")
	req.Header.Add("Sec-Ch-Ua", "Not;A=Brand;v=\"99\", Chromium;v=\"106\"")
	req.Header.Add("Accept", "application.json, text/plain, */*")
	req.Header.Add("Sec-Ch-Ua-Mobile", "?0")
	req.Header.Add("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/106.0.5249.62 Safari/537.36")
	req.Header.Add("Sec-Ch-Ua-Platform", "\"Windows\"")
	req.Header.Add("Origin", "https://www.retoactinver.com")
	req.Header.Add("Sec-Fetch-Site", "same-origin")
	req.Header.Add("Sec-Fetch-Mode", "cors")
	req.Header.Add("Sec-Fetch-Dest", "empty")
	req.Header.Add("Referer", "https://www.retoactinver.com/RetoActinver/")
	req.Header.Add("Accept-Encoding", "gzip, deflate")
	req.Header.Add("Accept-Language", "en-US,en;q=0.9")
	req.Header.Add("Connection", "close")

	// Agrega las cookies necesarias
	req.AddCookie(&http.Cookie{Name: "f5avraaaaaaaaaaaaaaaa_session_", Value: sessionInfo.Tokensesion})
	req.AddCookie(&http.Cookie{Name: "tokenapp", Value: sessionInfo.TokenApp})

	// Agrega los parámetros de la solicitud
	q := req.URL.Query()
	q.Add("cveUsuario", "osvaldo.hdz.m@outlook.com")
	q.Add("cx_tokenSesionApl", sessionInfo.Tokensesion)
	q.Add("cx_token_app", sessionInfo.TokenApp)
	q.Add("idRespuesta", strconv.Itoa(idRespuesta))
	req.URL.RawQuery = q.Encode()

	// Realiza la solicitud
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, err
	}

	// Devuelve la respuesta
	return resp, nil
}


func establishSession() {
    // Ejecutar la primera llamada curl y almacenar la salida en una variable
    output, err := exec.Command("curl", "-k", "-s", "-c", "-", "https://www.retoactinver.com/minisitio/reto/login.html").Output()
    if err != nil {
        fmt.Println("Error en la primera llamada a curl:", err)
        return
    }

    // Buscar el valor deseado en la salida de la llamada a curl
    cookie := ""
    lines := strings.Split(string(output), "\n")
    for _, line := range lines {
        if strings.Contains(line, "TS016e21d6") {
            parts := strings.Split(line, "\t")
            if len(parts) >= 2 {
                cookie = parts[1]
                break
            }
        }
    }

    if cookie == "" {
        fmt.Println("No se encontró el valor de TS016e21d6")
        return
    }

    // Crear un archivo JSON con el valor de TS016e21d6
    jsonContent := fmt.Sprintf("{ \"TS016e21d6\" : \"%s\" }\n", cookie)
    err = ioutil.WriteFile("SessionInfoTmp01.json", []byte(jsonContent), 0644)
    if err != nil {
        fmt.Println("Error al escribir SessionInfoTmp01.json:", err)
        return
    }

    // Ejecutar la segunda llamada curl
    postData := "{\"usuario\":\"osvaldo.hdz.m@outlook.com\",\"password\":\"Os23valdo1.\"}"
    cmd := exec.Command("curl", "-s", "-X", "POST", "-H", "Host: www.retoactinver.com", "-H", "Content-Length: 64", "-H", "Sec-Ch-Ua: Not;A=Brand;v=99, Chromium;v=106", "-H", "Accept: application/json, text/javascript, */*; q=0.01", "-H", "Content-Type: application/json", "-H", "X-Requested-With: XMLHttpRequest", "-H", "Sec-Ch-Ua-Mobile: ?0", "-H", "User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/106.0.5249.62 Safari/537.36", "-H", "Sec-Ch-Ua-Platform: Windows", "-H", "Origin: https://www.retoactinver.com", "-H", "Sec-Fetch-Site: same-origin", "-H", "Sec-Fetch-Mode: cors", "-H", "Sec-Fetch-Dest: empty", "-H", "Referer: https://www.retoactinver.com/minisitio/reto/login.html", "-H", "Accept-Encoding: gzip, deflate", "-H", "Accept-Language: es-419,es;q=0.9", "-H", "Connection: close", "-b", fmt.Sprintf("TS016e21d6=%s", cookie), "--data-binary", postData, "https://www.retoactinver.com/reto/app/usuarios/login")
    err = cmd.Run()
    if err != nil {
        fmt.Println("Error en la segunda llamada a curl:", err)
        return
    }

    // Ejecutar jq para combinar los archivos JSON
    cmd = exec.Command("jq", "-s", ".[0] * .[1]", "SessionInfoTmp01.json", "SessionInfoTmp02.json")
    err = cmd.Run()
    if err != nil {
        fmt.Println("Error al combinar los archivos JSON con jq:", err)
        return
    }
}


func getLoginInfo() {
    url := "https://www.retoactinver.com/reto/app/usuarios/login"

    payload := []byte(`{"usuario":"osvaldo.hdz.m@outlook.com","password":"Os23valdo1."}`)

    client := &http.Client{}
    req, err := http.NewRequest("POST", url, bytes.NewBuffer(payload))
    if err != nil {
        fmt.Println("Error al crear la solicitud HTTP:", err)
        return
    }

    // Configurar las cabeceras de la solicitud
    req.Header.Set("Host", "www.retoactinver.com")
    req.Header.Set("Content-Length", "64")
    req.Header.Set("Accept", "application/json")
    req.Header.Set("Content-Type", "application/json")
    req.Header.Set("Sec-Ch-Ua-Mobile", "?0")
    req.Header.Set("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.5845.111 Safari/537.36")
    req.Header.Set("Origin", "https://www.retoactinver.com")
    req.Header.Set("Sec-Fetch-Site", "same-origin")
    req.Header.Set("Sec-Fetch-Mode", "cors")
    req.Header.Set("Sec-Fetch-Dest", "empty")
    req.Header.Set("Referer", "https://www.retoactinver.com/minisitio/reto/login/index.html")
    req.Header.Set("Accept-Encoding", "gzip, deflate")
    req.Header.Set("Accept-Language", "en-US,en;q=0.9")
    req.Header.Set("Connection", "close")

    // Configurar las cookies
    cookies := "f5avraaaaaaaaaaaaaaaa_session_=GFJLNLIJNGACOIHCMBPFLJFOOCJGBANKGMAJCAPIEOHHPCELNIGMGDGJIEJGLBMKNAEDMFGJKGDPPLEAJEPANIMMPEAKCEEFPNOBNOJBMAJLCNCKIMPPPGIILCOHDKPH; _gid=GA1.2.1719040634.1696984986; _hjSessionUser_3213375=eyJpZCI6ImI8..."; // Las cookies completas aquí
    req.Header.Set("Cookie", cookies)

    resp, err := client.Do(req)
    if err != nil {
        fmt.Println("Error al realizar la solicitud HTTP:", err)
        return
    }
    defer resp.Body.Close()

    body, err := ioutil.ReadAll(resp.Body)
    if err != nil {
        fmt.Println("Error al leer la respuesta HTTP:", err)
        return
    }

    // Guardar la respuesta en un archivo JSON
    err = ioutil.WriteFile("LoginInfo.json", body, 0644)
    if err != nil {
        fmt.Println("Error al guardar la respuesta en un archivo JSON:", err)
        return
    }

	fmt.Println("\nRespuesta del servidor:")
    fmt.Println(string(body))

    fmt.Println("\nRespuesta guardada en LoginInfo.json")
}


func doTestRequest() {
	fmt.Printf("Realizando prueba de conexión...\n")

	resp, err := http.Get("https://www.retoactinver.com/RetoActinver/#/inicio")
	if err != nil {
		fmt.Println("Error:", err)
		return
	}
	defer resp.Body.Close()

	body, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		fmt.Println("Error al leer la respuesta:", err)
		return
	}

	fmt.Println("Respuesta:")
	fmt.Println(string(body))
}

func showHelp() {
	fmt.Println("Solve Daily Quizz - Ayuda")
	fmt.Println("Opciones:")
	flag.PrintDefaults()
}

