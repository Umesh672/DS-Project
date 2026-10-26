from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib


# ============================================================
# APPLICATION
# ============================================================

app = Flask(__name__)


# ============================================================
# LOAD MODEL
# ============================================================

model = joblib.load(
    "models/churn_model.pkl"
)


# ============================================================
# MODEL SCHEMA
# ============================================================

EXPECTED_COLUMNS = (

    model
    .named_steps[
        "preprocessor"
    ]
    .feature_names_in_
    .tolist()
)


# ============================================================
# EXPECTED TYPES
# ============================================================

NUMERIC_COLUMNS = [

    "SeniorCitizen",

    "tenure",

    "MonthlyCharges",

    "TotalCharges",

    "estimated_lifetime_value"
]


CATEGORICAL_COLUMNS = [

    "gender",

    "Partner",

    "Dependents",

    "PhoneService",

    "MultipleLines",

    "InternetService",

    "OnlineSecurity",

    "OnlineBackup",

    "DeviceProtection",

    "TechSupport",

    "StreamingTV",

    "StreamingMovies",

    "PaperlessBilling",

    "PaymentMethod",

    "Contract",

    "tenure_group"
]


# ============================================================
# VALIDATION
# ============================================================

def validate_input(data):

    errors = {

        "missing": [],

        "extra": [],

        "wrong_types": [],

        "null_columns": []
    }


    actual_columns = list(
        data.columns
    )


    expected_set = set(
        EXPECTED_COLUMNS
    )

    actual_set = set(
        actual_columns
    )


    # Missing

    errors["missing"] = sorted(

        expected_set
        -
        actual_set
    )


    # Extra

    errors["extra"] = sorted(

        actual_set
        -
        expected_set
    )


    # --------------------------------------------------------
    # Missing / extra
    # --------------------------------------------------------

    if (
        errors["missing"]
        or
        errors["extra"]
    ):

        return False, errors


    # --------------------------------------------------------
    # Reorder
    # --------------------------------------------------------

    data = data[
        EXPECTED_COLUMNS
    ]


    # --------------------------------------------------------
    # Null check
    # --------------------------------------------------------

    errors["null_columns"] = (

        data
        .columns[
            data.isnull().any()
        ]
        .tolist()
    )


    # --------------------------------------------------------
    # Numeric validation
    # --------------------------------------------------------

    for column in NUMERIC_COLUMNS:

        if column not in data.columns:

            continue


        if not pd.api.types.is_numeric_dtype(
            data[column]
        ):

            errors[
                "wrong_types"
            ].append({

                "column": column,

                "expected": "numeric",

                "actual":
                    str(
                        data[
                            column
                        ].dtype
                    )
            })


    # --------------------------------------------------------
    # Categorical validation
    # --------------------------------------------------------

    for column in CATEGORICAL_COLUMNS:

        if column not in data.columns:

            continue


        if not (

            pd.api.types.is_object_dtype(
                data[column]
            )

            or

            pd.api.types.is_string_dtype(
                data[column]
            )

        ):

            errors[
                "wrong_types"
            ].append({

                "column": column,

                "expected":
                    "string/categorical",

                "actual":
                    str(
                        data[
                            column
                        ].dtype
                    )
            })


    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    valid = not any([

        errors["missing"],

        errors["extra"],

        errors["wrong_types"],

        errors["null_columns"]

    ])


    return valid, errors


# ============================================================
# HOME
# ============================================================

@app.route(
    "/",
    methods=["GET"]
)
def home():

    return jsonify({

        "application":
            "Customer Churn Prediction API",

        "status":
            "running",

        "model":
            "Random Forest",

        "endpoint":
            "/predict"
    })


# ============================================================
# SCHEMA ENDPOINT
# ============================================================

@app.route(
    "/schema",
    methods=["GET"]
)
def schema():

    return jsonify({

        "expected_columns":
            EXPECTED_COLUMNS,

        "numeric_columns":
            NUMERIC_COLUMNS,

        "categorical_columns":
            CATEGORICAL_COLUMNS
    })


# ============================================================
# PREDICT
# ============================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    try:

        # ----------------------------------------------------
        # Read JSON
        # ----------------------------------------------------

        json_data = request.get_json()

        if not json_data:

            return jsonify({

                "success": False,

                "error":
                    "No JSON data supplied."

            }), 400


        # ----------------------------------------------------
        # Create DataFrame
        # ----------------------------------------------------

        input_data = pd.DataFrame(
            [json_data]
        )


        # ----------------------------------------------------
        # Feature engineering
        # ----------------------------------------------------

        if "tenure" in input_data.columns:

            tenure = float(
                input_data[
                    "tenure"
                ].iloc[0]
            )


            if tenure <= 12:

                tenure_group = (
                    "0-1 Year"
                )

            elif tenure <= 24:

                tenure_group = (
                    "1-2 Years"
                )

            elif tenure <= 48:

                tenure_group = (
                    "2-4 Years"
                )

            elif tenure <= 72:

                tenure_group = (
                    "4-6 Years"
                )

            else:

                tenure_group = (
                    "6+ Years"
                )


            input_data[
                "tenure_group"
            ] = tenure_group


        if (
            "MonthlyCharges"
            in input_data.columns

            and

            "tenure"
            in input_data.columns
        ):

            input_data[
                "estimated_lifetime_value"
            ] = (

                pd.to_numeric(
                    input_data[
                        "MonthlyCharges"
                    ],
                    errors="coerce"
                )

                *

                pd.to_numeric(
                    input_data[
                        "tenure"
                    ],
                    errors="coerce"
                )
            )


        # ----------------------------------------------------
        # Validate
        # ----------------------------------------------------

        valid, errors = (
            validate_input(
                input_data
            )
        )


        if not valid:

            return jsonify({

                "success": False,

                "message":
                    "Input validation failed.",

                "errors":
                    errors

            }), 400


        # ----------------------------------------------------
        # Reorder
        # ----------------------------------------------------

        input_data = input_data[
            EXPECTED_COLUMNS
        ]


        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        prediction = model.predict(
            input_data
        )[0]


        probability = (
            model
            .predict_proba(
                input_data
            )[0][1]
        )


        # ----------------------------------------------------
        # Risk
        # ----------------------------------------------------

        if probability >= 0.75:

            risk = "Very High"

        elif probability >= 0.50:

            risk = "High"

        elif probability >= 0.30:

            risk = "Medium"

        else:

            risk = "Low"


        # ----------------------------------------------------
        # Response
        # ----------------------------------------------------

        return jsonify({

            "success": True,

            "prediction":
                int(prediction),

            "churn":
                "Yes"
                if prediction == 1
                else "No",

            "churn_probability":
                round(
                    float(
                        probability
                    ),
                    4
                ),

            "risk":
                risk
        })


    except Exception as e:

        return jsonify({

            "success": False,

            "error":
                str(e)

        }), 500


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    app.run(

        host="0.0.0.0",

        port=5000,

        debug=True
    )