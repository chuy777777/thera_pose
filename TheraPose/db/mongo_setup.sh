#!/bin/bash

MONGODB1=mongodb1
MONGODB2=mongodb2
MONGODB3=mongodb3

echo "**********************************************" ${MONGODB1}
echo "Waiting for startup.."
sleep 30
echo "done"

mongo --host ${MONGODB1}:27017 -u ${MONGO_INITDB_ROOT_USERNAME} -p ${MONGO_INITDB_ROOT_PASSWORD} <<EOF
rsconf = {
    _id : "${MONGO_REPLICA_SET_NAME}",
    members: [
        {
            "_id": 0,
            "host": "${MONGODB1}:27017",
            "priority": 4
        },
        {
            "_id": 1,
            "host": "${MONGODB2}:27017",
            "priority": 2
        },
        {
            "_id": 2,
            "host": "${MONGODB3}:27017",
            "priority": 1
        }
    ]
};
rs.initiate(rsconf, { force: true });
EOF