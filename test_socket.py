import socketio
import time

sio = socketio.Client()

@sio.event
def connect():
    print("Connected")
    sio.emit('start_training', {})
    print("Emitted start_training")

@sio.on('training_update')
def on_message(data):
    print('Update:', data)

sio.connect('http://localhost:5001')
time.sleep(5)
sio.disconnect()
