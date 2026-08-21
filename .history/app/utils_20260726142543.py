import pandas as pd

# Model feature order
MODEL_FEATURES = [
    "absences",
    "failures",
    "goout",
    "age",
    "Mjob",
    "health",
    "Medu",
    "freetime",
    "Fedu",
    "Fjob",
    "Walc",
    "famrel",
    "studytime",
    "reason",
    "guardian",
    "Dalc",
    "schoolsup",
    "romantic",
    "traveltime",
    "activities",
    "paid",
    "sex",
    "famsize",
    "address",
    "famsup",
    "nursery",
    "higher",
    "internet"
]

# Label encodings used during training
ENCODINGS = {

    "sex":{
        "F":0,
        "M":1
    },

    "address":{
        "R":0,
        "U":1
    },

    "famsize":{
        "GT3":0,
        "LE3":1
    },

    "Mjob":{
        "at_home":0,
        "health":1,
        "other":2,
        "services":3,
        "teacher":4
    },

    "Fjob":{
        "at_home":0,
        "health":1,
        "other":2,
        "services":3,
        "teacher":4
    },

    "reason":{
        "course":0,
        "home":1,
        "other":2,
        "reputation":3
    },

    "guardian":{
        "father":0,
        "mother":1,
        "other":2
    },

    "schoolsup":{
        "no":0,
        "yes":1
    },

    "famsup":{
        "no":0,
        "yes":1
    },

    "paid":{
        "no":0,
        "yes":1
    },

    "activities":{
        "no":0,
        "yes":1
    },

    "nursery":{
        "no":0,
        "yes":1
    },

    "higher":{
        "no":0,
        "yes":1
    },

    "internet":{
        "no":0,
        "yes":1
    },

    "romantic":{
        "no":0,
        "yes":1
    }

}


def preprocess(data):

    df = pd.DataFrame([data])

    for column,mapping in ENCODINGS.items():
        df[column]=df[column].map(mapping)

    return df[MODEL_FEATURES]