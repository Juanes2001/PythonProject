import threading
import time
from nt import read

import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import serial
import numpy as np

import serial, serial.tools.list_ports
from serial.serialutil import SerialException

serialESP32 = None

def responsivity(x,y):
    if (x<200):
        return 0*y
    elif (x>=200 and x<475):
        return 2.5 * y
    # elif (x>=425 and x<450):
    # return y/(0.003*x-1.225)
    elif (x>=475 and x<500):
        return y/(0.0145*x-6.4)
    elif (x>=500 and x<512.5):
        return y/(0.004*x-1.15)
    elif (x>=512.5 and x<550):
        return y/(-0.0027*x+2.27)
    elif (x>=550 and x<575):
        return y/(0.008*x-3.6)
    elif (x>=575 and x<625):
        return y/(-0.006*x+4.45)
    elif (x>=625 and x<650):
        return y/(-0.0087*x+6.12)
    else:
        return 2.5 * y

def start_com ():
    global serialESP32;
    counter= 0
    # Iniciamos la comunicaci ́on serial con el COM y el BAUDRATE
    serialESP32 = serial.Serial("COM6", 115200, timeout = 1)
    print("Estableciendo comunicacion")
    while counter != 3:
            print(".")
            counter+=1
            time.sleep(1)# Dejamos un tiempo de demora de 1 segundo
    comando_de_inicio = ""

    #Limpiamos el buffer de todo mensaje basura que tenga disponible
    while serialESP32.in_waiting >0 :
        comando_de_inicio += str(serialESP32.readline().decode('utf-8')) #Vamos a leer lo mandado por el ESP

    comando_de_inicio = ""
    time.sleep(2)

    # Mandamos un mensaje de inicializacion por serial para que nos devuelva el menu de opciones
    while serialESP32.in_waiting > 0:
        comando_de_inicio += str(serialESP32.readline().decode('utf-8'))  # Vamos a leer lo mandado por el ESP

    comando_de_inicio = comando_de_inicio.replace("\r", "").replace("\n", "")
    if (comando_de_inicio == ">> SISTEMA INICIALIZADO"):
        print("Comunicación Establecida.."+ comando_de_inicio +"\n")
        comando_de_inicio = ""

        serialESP32.write("INICIO @".encode())
        time.sleep(3)
        while serialESP32.in_waiting > 0:
            comando_de_inicio += str(serialESP32.readline().decode('utf-8'))# Vamos a leer lo mandado por el ESP

        comando_de_inicio = comando_de_inicio.replace("\r", "")
        print(comando_de_inicio)

def plot(fig):
    global outListx
    global outListy
    global readState
    readState = 1
    while (1) and readState:
        datay = np.array(outListy[1:])
        # datax = np.array(outListx[1:])
        datax =  3./4. * (np.array(outListx[1:])-690) + 400
        for i in range(len(datay)):
             datay[i] = responsivity(datax[i],datay[i])
        fig.data[0].y = datay[:]
        fig.data[0].x = datax[:]
        time.sleep(1)

def read_countinuously():
    global outListx
    global outListy
    global readState
    global serialESP32
    global measurement_Done
    global end_programm
    readState = 1
    measurement_Done = 1
    line = ""
    comando_de_inicio = ""
    char_t = ""
    char_tpast = "ok"
    while (1) and readState:
        if serialESP32.in_waiting > 0 :
            line = str(serialESP32.readline().decode('utf-8')) # Lee los datos en cada línea
            line = line.replace("\r","")

        if "TERMINADO" in line :
            readState = 0 # Se termino la secuencia de lectura. por lo que el ploteo finalizará
            print (line + "..... REGRESANDO A CASA \n")
            break
        char_t = line # Decodificamos cada lınea
        try:
            char_t = char_t.split(" ")
            char_t[1] = char_t[1][:-1]
            char_t[0] = float(char_t[0])
            char_t[1] = float(char_t[1])

            if char_tpast != char_t[1]:
                outListx.append(char_t[1])
                outListy.append(char_t[0])
            char_tpast = char_t[1]
        except:
            continue
    while serialESP32.in_waiting > 0 or not ("--||--" in comando_de_inicio):
        comando_de_inicio += str(serialESP32.readline().decode('utf-8'))  # Vamos a leer lo mandado por el PUERTO

    comando_de_inicio = comando_de_inicio.replace("--||--", "")

    print(comando_de_inicio)
    measurement_Done = 0
    end_programm = 1

def send_receive_command(comando):
    global serialESP32
    if '@' in comando:
        serialESP32.write(comando.encode())
        time.sleep(3)
        comando_de_inicio = ""
        while serialESP32.in_waiting > 0 :
            comando_de_inicio += str(serialESP32.readline().decode('utf-8'))  # Vamos a leer lo mandado por el PUERTO

        comando_de_inicio = comando_de_inicio.replace("\r", "")
        if "--||--" in comando_de_inicio:
            comando_de_inicio = comando_de_inicio.replace("--||--", "")
            print(comando_de_inicio)
            if not "Tarea desconocida...Corrija su comando" in comando_de_inicio:
                if "step_motor" in comando:
                    serialESP32.write("RECIBIDO_step_motor @".encode())
                    comando_de_inicio = ""
                    while serialESP32.in_waiting > 0 or not ("--||--" in comando_de_inicio):
                        comando_de_inicio += str(serialESP32.readline().decode('utf-8'))  # Vamos a leer lo mandado por el PUERTO

                    comando_de_inicio = comando_de_inicio.replace("--||--", "")

                    print(comando_de_inicio)
                elif "start_measurements" in comando:
                    serialESP32.write("RECIBIDO_start_measurements @".encode())
                    readThread.start()
                    plotThread.start() #Comenzamos los hilos que graficaran en tiempo real lo que le llegue al puerto
                elif "on_lux" in comando:
                    serialESP32.write("RECIBIDO_on_lux @".encode())
                    comando_de_inicio = ""
                    while not "--||--" in comando_de_inicio:
                        comando_de_inicio = ""
                        while serialESP32.in_waiting > 0 or not ("MANDADO" in comando_de_inicio):
                            comando_de_inicio += str(serialESP32.readline().decode('utf-8'))  # Vamos a leer lo mandado por el PUERTO
                        if not "--||--" in comando_de_inicio:
                            comando_de_inicio = comando_de_inicio.replace("\r", "").replace("MANDADO", "")
                            print(comando_de_inicio)
                            serialESP32.write("RECIBIDO_on_lux @".encode())
                        else:
                            comando_de_inicio = comando_de_inicio.replace("\r", "").replace("MANDADO", "")

                    comando_de_inicio = comando_de_inicio.replace("--||--", "")
                    print(comando_de_inicio)


    else:
        print("\nCOMANDO INCORRECTO, RECUERDE LA TERMINACION @ AL FINAL DEL MISMO\n")


if __name__ == "__main__" :
    layout = go.Layout(
        title="Espectro LED Blanco",
        plot_bgcolor="#FFFFFF",
        hovermode="x",
        hoverdistance   = 100,  # Distance to show hover label of data point
        spikedistance   = 1000,  # Distance to show spike
        xaxis=dict(
            title="Longitud de onda (nm)",
            linecolor="#BCCCDC",
            showspikes=True,  # Show spike line for X-axis
            # Format spike
            spikethickness=2,
            spikedash="dot",
            spikecolor="#999999",
            spikemode="across",
        ),

        yaxis=dict(
            title="Intensidad Relativa (lux)",
            linecolor="#BCCCDC"
        )
    )
    fig = go.FigureWidget(layout=layout)
    fig.add_scatter()
    lastLen = 0
    outListx = [""]
    outListy = [""]
    readState = 0
    measurement_Done = 0
    end_programm = 0


    readThread = threading.Thread(name='Read', target=read_countinuously)
    plotThread = threading.Thread(name='Plot', target=plot, args=(fig,))
    start_com()
    while(1):
        #Aqui comenzamos nuestro programa infinitamente
        command = input("\nInserte el comando.. ")
        send_receive_command(command)

        while measurement_Done:
            pass

        if end_programm:
            serialESP32.close()
            time.sleep(2)
            print("REINICIE EL PROGRAMA SI QUIERE USAR DE NUEVO LA FUNCION START_MEASUREMENTS \n")
            print ("PROGRAMA TERMINADO\n")
            fig.show()
            break

