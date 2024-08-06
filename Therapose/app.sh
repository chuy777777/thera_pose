#!/bin/bash 

# Al crear el archivo se le dieron permisos de ejecucion:
#   - chmod 777 app.sh 

# Funcion que imprime el nombre de la aplicacion
# figlet App (de aqui se optiene el texto para el logo)
app_name="therapose"

function print_logo(){
    figlet $app_name
    echo ""
}

# Funcion que imprime los 'logs' de los servicios
function print_logs(){
    service=$1
    tail=$2
    service_name=$([ "$service" = "db" ] && echo "Base de datos" || echo "Aplicacion")
    echo "  - $service_name (logs):"
    while IFS= read -r line
    do
        echo -e "\t$line"
    done < <(docker-compose -f ./$service/docker-compose.yaml logs --tail $([ "$tail" = "" ] && echo "all" || echo $tail))
    echo ""
}

# Funcion que imprime el estado actual de los servicios
function print_status(){
    echo "Estado actual de los servicios:"
    for service in "db" "app" 
    do
        service_name=$([ "$service" = "db" ] && echo "Base de datos" || echo "Aplicacion")
        res=$(is_active $service)
        if [ "$res" -eq 0 ]; then
            # Servicio detenido
            echo -e "\033[31m  - '$service_name' detenido \033[0m"
        else
            # Servicio iniciado
            echo -e "\033[32m  - '$service_name' iniciado \033[0m"
        fi
    done
    echo ""
}

# Funcion que verifica el estado de los servicios
#   - Base de datos
#   - Aplicacion 
function is_active(){
    count=0
    while IFS=: read -r elem ; 
    do
        ((count++))
    done < <(docker compose -f ./$1/docker-compose.yaml ps --format json)
    
    if [ "$count" -eq 0 ]; then
        # Esta detenido el servicio
        echo 0
    else
        # Esta iniciado el servicio
        echo 1
    fi
}

# Funcion para confirmar una accion critica
function ask_yes_no(){
    read -e -p "$1 (si/no)? " sino
    echo "$sino"
}

# Funcion para crear una red en Docker
function create_network(){
    network_exists=false
    while IFS= read -r network
    do
        if [ "$network" == "network_$app_name" ]; then
            network_exists=true
        fi
    done < <(docker network ls | awk '{print $2}' | tail -n+2)
    if ! $network_exists; then
        res=$(docker network create -d bridge network_$app_name)
    fi
}

# Funcion para eliminar contenedores con 'status=exited'
function delete_exited_containers(){
    while IFS= read -r container_id
    do
        res=$(docker rm -f $container_id)
    done < <(docker ps -a -f status=exited -q)
}

delete_exited_containers
clear
print_logo
print_status

# Si no existe la red 'network_$app_name' la creamos 
# Esta red permite la comunicacion entre servicios 
create_network
                                                                                                                                               
option_1="Salir"
option_2="Iniciar/Detener base de datos (local)"
option_3="Iniciar/Detener aplicacion"
option_4="Recargar"
option_5="Mostrar 'logs'"
option_6="Eliminar volumenes de la base de datos"
option_7="Eliminar volumenes de la aplicacion"

select option in "$option_1" "$option_2" "$option_3" "$option_4" "$option_5" "$option_6" "$option_7" ;
do
    logs=false

    if [ "$option" = "$option_1" ]; then
        exit
    elif [ "$option" = "$option_2" ]; then
        if [ -f "./db/mongo_replication.key" ]; then  
            # Si existe el archivo verificamos que el comando 'chown' se haya ejecutado correctamente 
            file_owner=$(ls -l ./db/mongo_replication.key | awk '{print $3}')
            file_group=$(ls -l ./db/mongo_replication.key | awk '{print $4}')
            owner=$(id -nu 999)
            group=$(id -ng 999)
            if [[ ! "$file_owner" = "$owner" ]] && [[ ! "$file_group" = "$group" ]]; then
                rm -f ./db/mongo_replication.key
            fi
        fi

        # Iniciamos/Detenemos el servicio de base de datos
        res=$(is_active "db")
        if [ "$res" -eq 0 ]; then
            if ! [ -f "./db/mongo_replication.key" ]; then  
                # Si no existe el archivo entonces lo creamos
                openssl rand -base64 768 > ./db/mongo_replication.key
                chmod 400 ./db/mongo_replication.key
                # Se crea un bucle hasta que el usuario introduzca la contrasena correcta
                sudo chown 999:999 ./db/mongo_replication.key
            fi

            # Iniciamos el servicio de base de datos
            docker-compose -f ./db/docker-compose.yaml --env-file ./db/.env up -d
        else
            # Detenemos el servicio de base de datos
            docker-compose -f ./db/docker-compose.yaml --env-file ./db/.env down
        fi
    elif [ "$option" = "$option_3" ]; then
        # Este comando es necesario para dar permisos de que Docker abra interfaces graficas
        res=$(xhost +local:docker)
        # Iniciamos/Detenemos el servicio de aplicacion
        res=$(is_active "app")
        if [ "$res" -eq 0 ]; then
            # Iniciamos el servicio de aplicacion
            docker-compose -f ./app/docker-compose.yaml up -d
        else
            # Detenemos el servicio de aplicacion
            docker-compose -f ./app/docker-compose.yaml down
        fi
    elif [ "$option" = "$option_4" ]; then
        :
    elif [ "$option" = "$option_5" ]; then
        logs=true
    elif [ "$option" = "$option_6" ]; then
        while IFS= read -r sino
        do
            if [ "$sino" = "si" ]; then
                res=$(docker volume rm volume_mongodb1 volume_mongodb2 volume_mongodb3)
                # echo "Eliminacion satisfactoria."
            fi
        done < <(ask_yes_no "Desea eliminar los volumenes de la base de datos")
    elif [ "$option" = "$option_7" ]; then
        while IFS= read -r sino
        do
            if [ "$sino" = "si" ]; then
                res=$(docker volume rm volume_$app_name)
                # echo "Eliminacion satisfactoria."
            fi
        done < <(ask_yes_no "Desea eliminar los volumenes de la aplicacion")
    else 
        echo "Opcion invalida"
    fi 

    delete_exited_containers
    print_logo
    if $logs; then
        print_logs "db" "1"
        print_logs "app" "10"
    fi
    print_status

    REPLY=
done
