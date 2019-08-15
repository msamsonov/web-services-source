docker exec -it flask-hello_flask_1 bash

curl --header "Content-Type: application/json" --request POST --data '{"username": "xyz", "password": "xyz"}' http://localhost:5000/iris_post