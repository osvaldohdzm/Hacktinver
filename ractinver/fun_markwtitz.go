package main

import (
    "fmt"
    "gonum.org/v1/gonum/mat"
    "gonum.org/v1/gonum/optimize"
    "gonum.org/v1/gonum/optimize/functions"
)

func main() {
    // Definir los datos históricos de rendimiento y la matriz de covarianza (simulados)
    returns := []float64{0.1, 0.15, 0.12}
    covMatrix := mat.NewSymDense(3, nil, []float64{0.05, 0.03, 0.02, 0.03, 0.08, 0.05, 0.02, 0.05, 0.06})

    // Definir la función objetivo (varianza de la cartera)
    f := func(x []float64) float64 {
        var xVec mat.VecDense
        xVec.Append(x)
        var xTranspose mat.VecDense
        xTranspose.MulElem(&xVec, &xVec)
        var portfolioRisk mat.VecDense
        portfolioRisk.MulElem(covMatrix, &xTranspose)
        return mat.Sum(&portfolioRisk)
    }

    // Definir restricciones (la suma de los pesos debe ser igual a 1)
    constraints := []optimize.Constraint{
        {
            F: func(x []float64) float64 {
                sum := 0.0
                for _, w := range x {
                    sum += w
                }
                return 1.0 - sum
            },
            Grad: func(grad, x []float64) {
                for i := range grad {
                    grad[i] = -1.0
                }
            },
        },
    }

    // Configurar opciones de optimización
    settings := optimize.DefaultSettings()
    settings.Recorder = nil // Deshabilitar la grabación para simplificar la salida

    // Crear un problema de optimización
    problem := optimize.Problem{
        Func: f,
        Constraints: constraints,
    }

    // Realizar la optimización
    result, err := optimize.Minimize(problem, []float64{1.0, 0.0, 0.0}, &settings, nil)
    if err != nil {
        fmt.Println(err)
        return
    }

    // Imprimir los resultados
    fmt.Printf("Peso óptimo de activos en la cartera: %v\n", result.X)
    fmt.Printf("Rendimiento esperado de la cartera: %.2f\n", -result.F)
}
