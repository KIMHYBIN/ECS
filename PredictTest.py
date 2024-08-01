import socket
import pandas as pd
from tensorflow.keras.models import load_model

host = "127.0.0.1"
port = 10001
servSock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
servSock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
servSock.bind((host, port))
servSock.listen()

while True:
    print("클라이언트 연결 대기 중...")
    clntSock, addr = servSock.accept()
    print(f"클라이언트 {addr} 연결 성공!")

    while True:
        print('*-*-*-*-*')
        data = clntSock.recv(1024)
        msg = data.decode().strip()
        print(msg)
        if not msg:
            print("클라이언트 연결 오류.")
            clntSock.close()
            break

        msg_parts = msg.split(',')
        print(f'msg_parts: ', msg_parts[0], msg_parts[1], msg_parts[2], msg_parts[3])
        Years = int(msg_parts[0])
        Location = int(msg_parts[1])
        AllPeople = int(msg_parts[2])


        if msg_parts[0] == '1':
            modelpath = "./data/model/Ch15-uncommon6.hdf5"
            loaded_model = load_model(modelpath)

            new_data = pd.DataFrame({'연도': [Years], '지역': [Location], '전체 인구 수': [AllPeople]})
            X_new = new_data
            PredictPeople = loaded_model.predict(X_new)

            print("장애인 수 예측:", PredictPeople[0][0])

            clntSock.sendall(str(PredictPeople[0][0]).encode())

        elif msg_parts[0] == '9':
            print("클라이언트 연결 종료.")
            clntSock.close()