var mongo_initdb_username = _getEnv("MONGO_INITDB_USERNAME")
var mongo_initdb_passwrod = _getEnv("MONGO_INITDB_PASSWORD")
var mongo_initdb_database = _getEnv("MONGO_INITDB_DATABASE")

admin = db.getSiblingDB("admin");
admin.createUser({
        user: mongo_initdb_username,
        pwd: mongo_initdb_passwrod,
        roles: [{ role: "readWrite", db: mongo_initdb_database }],
});

initdb = db.getSiblingDB(mongo_initdb_database);

function getRandomArbitrary(min, max) {
        return Number((Math.random() * (max - min) + min).toFixed(2));
}

const date = new Date();
var dateSplit = date.toLocaleDateString("es-MX").split("/");
var currentDay = dateSplit[0].padStart(2, "0");
var currentMonth = dateSplit[1].padStart(2, "0");
var currentYear = dateSplit[2];
var currentDateTime = `${currentYear}-${currentMonth}-${currentDay} ${date.toLocaleTimeString("es-MX")}`;

initdb.createCollection("DatabaseUpdates");     // MODIFICABLE
initdb.createCollection("Session");             // NO MODIFICABLE
initdb.createCollection("CurrentSituation");    // NO MODIFICABLE
initdb.createCollection("Patient");             // MODIFICABLE                             
initdb.createCollection("Disease");             // MODIFICABLE
initdb.createCollection("Pathology");           // MODIFICABLE

var resDatabaseUpdates = initdb.DatabaseUpdates.insertOne({
        "database_update_date": currentDateTime,
});
var resSession = initdb.Session.insertMany([
        {
                "name": "No especificado"
        },
        {
                "name": "Terapia"
        },
        {
                "name": "Valoracion"
        },
        {
                "name": "Masaje"
        }
]);
var resCurrentSituation = initdb.CurrentSituation.insertMany([
        {
                "name": "Activo",
                "color": [0, 255, 0]
        },
        {
                "name": "Inactivo",
                "color": [255, 0, 0]
        },
        {
                "name": "Dado de alta",
                "color": [0, 0, 255]
        },
        {
                "name": "Pendiente (no se sabe)",
                "color": [220, 220, 220]
        }
]);
var resPathology = initdb.Pathology.insertMany([
        {
                "name": "Hombro congelado"
        },
        {
                "name": "Meniscopatia"
        },
        {
                "name": "Condromalacia rotuliana"
        },
        {
                "name": "Gonartrosis"
        },
        {
                "name": "Desgarro tipo 1"
        },
        {
                "name": "Desgarro tipo 2"
        },
        {
                "name": "Desgarro tipo 3"
        },
        {
                "name": "Codo de golfista"
        },
        {
                "name": "Codo tenista"
        },
        {
                "name": "Hernia discal"
        },
        {
                "name": "Artrosis"
        },
        {
                "name": "Hemiplejia"
        },
        {
                "name": "Incontinencia urinaria"
        },
]);
var resDisease = initdb.Disease.insertMany([
        {
                "name": "Diabetes mellitus tipo 1"
        },
        {
                "name": "Diabetes mellitus tipo 2"
        },
        {
                "name": "Diabetes gestacional"
        },
        {
                "name": "Obesidad tipo 1"
        },
        {
                "name": "Obesidad tipo 2"
        },
        {
                "name": "Obesidad tipo 3"
        },
        {
                "name": "Sobrepeso"
        },
        {
                "name": "Hipertension arterial"
        },
        {
                "name": "Colesterol alto"
        },
        {
                "name": "Hipotension arterial "
        },
        {
                "name": "Sindrome metabolico"
        },
        {
                "name": "Cancer de mama"
        },
        {
                "name": "Cancer de prostata"
        },
        {
                "name": "Cancer de cervix uterino"
        },
]);

// Aqui se puede automatizar un proceso de creacion de pacientes
function getRandomInt(min, max) {
        return Math.floor(min + Math.random() * (max - min));
}

function getDateWithRandomTime(n = 0) {
        let date = new Date();
        if (n != 0) {
                date.setDate(date.getDate() + n)
        }
        var dateSplit = date.toISOString().slice(0, 10).replace(/-/g, " ").split(" ")
        var currentDay = dateSplit[2]
        var currentMonth = dateSplit[1]
        var currentYear = dateSplit[0];
        var randomTime = `${getRandomInt(7, 22).toString().padStart(2, '0')}:00:00`
        var currentDateTime = `${currentYear}-${currentMonth}-${currentDay} ${randomTime}`;
        return currentDateTime
}

function getConsecutiveDates(n) {
        let consecutiveDates = []
        for (let i = 0; i < n; i++) {
                consecutiveDates.push(getDateWithRandomTime(i))
        }
        return consecutiveDates
}


let names = ["Hugo", "Mateo", "Martin", "Lucas", "Leo", "Daniel", "Alejandro", "Manuel", "Pablo", "Alvaro", "Adrian", "Enzo", "Mario", "Diego", "David", "Oliver", "Marcos", "Thiago", "Marco", "Alex", "Javier", "Izan", "Bruno", "Miguel", "Antonio", "Gonzalo", "Liam", "Gael", "Marc", "Carlos", "Juan", "Angel", "Dylan", "Nicolas", "Jose", "Sergio", "Gabriel", "Luca", "Jorge", "Dario", "Iker", "Samuel", "Eric", "Adam", "Hector", "Francisco", "Rodrigo", "Jesus", "Erik", "Amir", "Jaime", "Ian", "Ruben", "Aaron", "Ivan", "Pau", "Victor", "Guillermo", "Luis", "Mohamed", "Pedro", "Julen", "Unai", "Rafael", "Santiago", "Saul", "Alberto", "Noah", "Aitor", "Joel", "Nil", "Jan", "Pol", "Raul", "Matias", "Marti", "Fernando", "Andres", "Rayan", "Alonso", "Ismael", "Asier", "Biel", "Ander", "Aleix", "Axel", "Alan", "Ignacio", "Fabio", "Neizan", "Jon", "Teo", "Isaac", "Arnau", "Luka", "Max", "Imran", "Youssef", "Anas", "Elias"]
let lastNames = ["Hernandez", "Garcia", "Martinez", "Lopez", "Gonzalez", "Perez", "Rodriguez", "Sanchez", "Ramirez", "Cruz", "Gomez", "Flores", "Morales", "Vazquez", "Jimenez", "Reyes", "Diaz", "Torres", "Gutierrez", "Ruiz", "Mendoza", "Aguilar", "Mendez", "Moreno", "Ortiz", "Juarez", "Castillo", "Alvarez", "Romero", "Ramos", "Rivera", "Chavez", "De la Cruz", "Dominguez", "Guzman", "Velazquez", "Santiago", "Herrera", "Castro", "Vargas", "Medina", "Rojas", "Muñoz", "Luna", "Contreras", "Bautista", "Salazar", "Ortega", "Guerrero", "Estrada"]
let patientList = []
for (let i = 0; i < 10; i++) {
        let name = names[getRandomInt(0, names.length)]
        let lastName = lastNames[getRandomInt(0, lastNames.length)]

        let fullName = `${name} ${lastName}`
        let session = resSession.insertedIds[getRandomInt(0, resSession.insertedIds.length)]
        let pathologies = resPathology.insertedIds.slice(0, getRandomInt(0, resPathology.insertedIds.length))
        let schedule = getConsecutiveDates(getRandomInt(0, 10))
        let diseases = resDisease.insertedIds.slice(0, getRandomInt(0, resDisease.insertedIds.length))
        let currentSituation = resCurrentSituation.insertedIds[getRandomInt(0, resCurrentSituation.insertedIds.length)]

        patientList.push({
                "name": fullName,
                "session": session,
                "pathologies": pathologies,
                "schedule": schedule,
                "diseases": diseases,
                "current_situation": currentSituation,
        })
}

var resPatient = initdb.Patient.insertMany(patientList)
