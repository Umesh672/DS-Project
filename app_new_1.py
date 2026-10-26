
# ============================================================
# CUSTOMER CHURN PREDICTION - STREAMLIT APP
# ============================================================

import warnings

warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# CONSTANTS
# ============================================================

MODEL_PATH = "models/churn_model.pkl"


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    return joblib.load(MODEL_PATH)


try:

    model = load_model()

except Exception as e:

    st.error(
        "❌ Unable to load the trained model."
    )

    st.exception(e)

    st.stop()


# ============================================================
# GET MODEL SCHEMA
# ============================================================

try:

    EXPECTED_COLUMNS = (
        model
        .named_steps["preprocessor"]
        .feature_names_in_
        .tolist()
    )

except Exception as e:

    st.error(
        "❌ Unable to read the model training schema."
    )

    st.exception(e)

    st.stop()


# ============================================================
# EXPECTED DATA TYPES
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
# VALIDATION FUNCTION
# ============================================================

def validate_input(dataframe):

    errors = {

        "missing": [],

        "extra": [],

        "wrong_types": [],

        "null_columns": [],

        "column_order": False
    }


    # --------------------------------------------------------
    # COLUMN CHECK
    # --------------------------------------------------------

    actual_columns = list(
        dataframe.columns
    )

    expected_set = set(
        EXPECTED_COLUMNS
    )

    actual_set = set(
        actual_columns
    )


    errors["missing"] = sorted(
        expected_set - actual_set
    )

    errors["extra"] = sorted(
        actual_set - expected_set
    )


    # --------------------------------------------------------
    # STOP IF MISSING OR EXTRA COLUMNS
    # --------------------------------------------------------

    if (
        errors["missing"]
        or
        errors["extra"]
    ):

        return False, errors


    # --------------------------------------------------------
    # COLUMN ORDER
    # --------------------------------------------------------

    if actual_columns != EXPECTED_COLUMNS:

        errors["column_order"] = True

        return False, errors


    # --------------------------------------------------------
    # NULL CHECK
    # --------------------------------------------------------

    errors["null_columns"] = (
        dataframe
        .columns[
            dataframe.isnull().any()
        ]
        .tolist()
    )


    # --------------------------------------------------------
    # NUMERIC TYPE CHECK
    # --------------------------------------------------------

    for column in NUMERIC_COLUMNS:

        if column not in dataframe.columns:

            continue

        if not pd.api.types.is_numeric_dtype(
            dataframe[column]
        ):

            errors["wrong_types"].append({

                "column": column,

                "expected": "numeric",

                "actual": str(
                    dataframe[column].dtype
                )
            })


    # --------------------------------------------------------
    # CATEGORICAL TYPE CHECK
    # --------------------------------------------------------

    for column in CATEGORICAL_COLUMNS:

        if column not in dataframe.columns:

            continue

        if not (
            pd.api.types.is_object_dtype(
                dataframe[column]
            )

            or

            pd.api.types.is_string_dtype(
                dataframe[column]
            )

            or

            isinstance(
                dataframe[column].dtype,
                pd.CategoricalDtype
            )
        ):

            errors["wrong_types"].append({

                "column": column,

                "expected": "string/categorical",

                "actual": str(
                    dataframe[column].dtype
                )
            })


    # --------------------------------------------------------
    # FINAL VALIDATION
    # --------------------------------------------------------

    has_errors = (

        len(errors["missing"]) > 0

        or

        len(errors["extra"]) > 0

        or

        len(errors["wrong_types"]) > 0

        or

        len(errors["null_columns"]) > 0

        or

        errors["column_order"]
    )


    return (
        not has_errors,
        errors
    )


# ============================================================
# TITLE
# ============================================================

st.title(
    "📊 Customer Churn Prediction System"
)

st.write(
    """
    Predict whether a customer is likely to churn
    using a trained Random Forest machine-learning model.
    """
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "👤 Customer Information"
)


# ============================================================
# CUSTOMER INFORMATION
# ============================================================

gender = st.sidebar.selectbox(
    "Gender",
    ["Male", "Female"]
)


senior_citizen = st.sidebar.selectbox(
    "Senior Citizen",
    [0, 1],
    format_func=lambda x:
        "Yes" if x == 1 else "No"
)


partner = st.sidebar.selectbox(
    "Partner",
    ["Yes", "No"]
)


dependents = st.sidebar.selectbox(
    "Dependents",
    ["Yes", "No"]
)


tenure = st.sidebar.number_input(
    "Tenure (Months)",
    min_value=0,
    max_value=100,
    value=12,
    step=1
)


phone_service = st.sidebar.selectbox(
    "Phone Service",
    ["Yes", "No"]
)


multiple_lines = st.sidebar.selectbox(
    "Multiple Lines",
    [
        "Yes",
        "No",
        "No phone service"
    ]
)


# ============================================================
# INTERNET SERVICES
# ============================================================

internet_service = st.sidebar.selectbox(
    "Internet Service",
    [
        "DSL",
        "Fiber optic",
        "No"
    ]
)


online_security = st.sidebar.selectbox(
    "Online Security",
    [
        "Yes",
        "No",
        "No internet service"
    ]
)


online_backup = st.sidebar.selectbox(
    "Online Backup",
    [
        "Yes",
        "No",
        "No internet service"
    ]
)


device_protection = st.sidebar.selectbox(
    "Device Protection",
    [
        "Yes",
        "No",
        "No internet service"
    ]
)


tech_support = st.sidebar.selectbox(
    "Tech Support",
    [
        "Yes",
        "No",
        "No internet service"
    ]
)


streaming_tv = st.sidebar.selectbox(
    "Streaming TV",
    [
        "Yes",
        "No",
        "No internet service"
    ]
)


streaming_movies = st.sidebar.selectbox(
    "Streaming Movies",
    [
        "Yes",
        "No",
        "No internet service"
    ]
)


# ============================================================
# BILLING
# ============================================================

paperless_billing = st.sidebar.selectbox(
    "Paperless Billing",
    ["Yes", "No"]
)


payment_method = st.sidebar.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer",
        "Credit card"
    ]
)


contract = st.sidebar.selectbox(
    "Contract",
    [
        "Month-to-month",
        "One year",
        "Two year"
    ]
)


monthly_charges = st.sidebar.number_input(
    "Monthly Charges",
    min_value=0.0,
    value=50.0,
    step=1.0
)


total_charges = st.sidebar.number_input(
    "Total Charges",
    min_value=0.0,
    value=600.0,
    step=10.0
)


# ============================================================
# FEATURE ENGINEERING
# ============================================================

if tenure <= 12:

    tenure_group = "0-1 Year"

elif tenure <= 24:

    tenure_group = "1-2 Years"

elif tenure <= 48:

    tenure_group = "2-4 Years"

elif tenure <= 72:

    tenure_group = "4-6 Years"

else:

    tenure_group = "6+ Years"


estimated_lifetime_value = (
    float(monthly_charges)
    *
    float(tenure)
)


# ============================================================
# CREATE INPUT DATAFRAME
# ============================================================

input_data = pd.DataFrame({

    "gender": [str(gender)],

    "SeniorCitizen": [
        int(senior_citizen)
    ],

    "Partner": [str(partner)],

    "Dependents": [
        str(dependents)
    ],

    "tenure": [
        int(tenure)
    ],

    "PhoneService": [
        str(phone_service)
    ],

    "MultipleLines": [
        str(multiple_lines)
    ],

    "InternetService": [
        str(internet_service)
    ],

    "OnlineSecurity": [
        str(online_security)
    ],

    "OnlineBackup": [
        str(online_backup)
    ],

    "DeviceProtection": [
        str(device_protection)
    ],

    "TechSupport": [
        str(tech_support)
    ],

    "StreamingTV": [
        str(streaming_tv)
    ],

    "StreamingMovies": [
        str(streaming_movies)
    ],

    "PaperlessBilling": [
        str(paperless_billing)
    ],

    "PaymentMethod": [
        str(payment_method)
    ],

    "Contract": [
        str(contract)
    ],

    "MonthlyCharges": [
        float(monthly_charges)
    ],

    "TotalCharges": [
        float(total_charges)
    ],

    "tenure_group": [
        str(tenure_group)
    ],

    "estimated_lifetime_value": [
        float(estimated_lifetime_value)
    ]
})


# ============================================================
# REORDER COLUMNS TO EXACT MODEL SCHEMA
# ============================================================

try:

    input_data = input_data[
        EXPECTED_COLUMNS
    ]

except KeyError as e:

    st.error(
        "❌ Model schema and application schema "
        "do not match."
    )

    st.exception(e)

    st.stop()


# ============================================================
# CUSTOMER INFORMATION DISPLAY
# ============================================================

col1, col2 = st.columns(
    [2, 1]
)


with col1:

    st.subheader(
        "👤 Customer Information"
    )

    # --------------------------------------------------------
    # IMPORTANT FIX:
    #
    # Do NOT use:
    #
    # input_data.T.rename(...)
    #
    # because the resulting Value column contains
    # strings + integers + floats.
    #
    # PyArrow may therefore raise:
    #
    # ArrowTypeError:
    # Expected bytes, got a 'int' object
    #
    # Convert everything to string for display only.
    # --------------------------------------------------------

    display_data = pd.DataFrame({

        "Feature": input_data.columns.astype(str),

        "Value": [
            str(value)
            for value in input_data.iloc[0].tolist()
        ]
    })


    st.dataframe(

        display_data,

        width="stretch",

        hide_index=True
    )


with col2:

    st.subheader(
        "📈 Customer Metrics"
    )

    st.metric(
        "Tenure",
        f"{tenure} months"
    )

    st.metric(
        "Monthly Charges",
        f"${float(monthly_charges):,.2f}"
    )

    st.metric(
        "Estimated Lifetime Value",
        f"${estimated_lifetime_value:,.2f}"
    )


st.divider()


# ============================================================
# PREDICTION BUTTON
# ============================================================

if st.button(

    "🔮 Predict Customer Churn",

    width="stretch"
):

    # --------------------------------------------------------
    # VALIDATE INPUT
    # --------------------------------------------------------

    is_valid, errors = validate_input(
        input_data
    )


    # --------------------------------------------------------
    # VALIDATION FAILED
    # --------------------------------------------------------

    if not is_valid:

        st.error(
            "❌ Input validation failed."
        )


        # ----------------------------------------------------
        # MISSING COLUMNS
        # ----------------------------------------------------

        if errors["missing"]:

            st.warning(
                "Missing columns:"
            )

            st.code(
                "\n".join(
                    errors["missing"]
                )
            )


        # ----------------------------------------------------
        # EXTRA COLUMNS
        # ----------------------------------------------------

        if errors["extra"]:

            st.warning(
                "Unexpected columns:"
            )

            st.code(
                "\n".join(
                    errors["extra"]
                )
            )


        # ----------------------------------------------------
        # WRONG DATA TYPES
        # ----------------------------------------------------

        if errors["wrong_types"]:

            st.warning(
                "Incorrect feature data types:"
            )

            for error in errors[
                "wrong_types"
            ]:

                st.write(

                    f"**{error['column']}** — "

                    f"Expected: "
                    f"`{error['expected']}` | "

                    f"Received: "
                    f"`{error['actual']}`"
                )


        # ----------------------------------------------------
        # NULL VALUES
        # ----------------------------------------------------

        if errors["null_columns"]:

            st.warning(
                "Missing values found in:"
            )

            st.code(
                "\n".join(
                    errors[
                        "null_columns"
                    ]
                )
            )


        # ----------------------------------------------------
        # COLUMN ORDER
        # ----------------------------------------------------

        if errors["column_order"]:

            st.warning(
                "Column order does not match "
                "the training schema."
            )


        st.stop()


    # ========================================================
    # PREDICTION
    # ========================================================

    try:

        prediction = model.predict(
            input_data
        )[0]


        probability = (
            model
            .predict_proba(
                input_data
            )[0][1]
        )


        probability = float(
            probability
        )


        # ====================================================
        # RESULT
        # ====================================================

        st.subheader(
            "🎯 Prediction Result"
        )


        result_col1, result_col2, result_col3 = (
            st.columns(3)
        )


        with result_col1:

            st.metric(

                "Churn Probability",

                f"{probability * 100:.2f}%"
            )


        with result_col2:

            if prediction == 1:

                st.error(
                    "⚠️ Customer Likely to Churn"
                )

            else:

                st.success(
                    "✅ Customer Likely to Stay"
                )


        with result_col3:

            st.metric(

                "Prediction",

                "CHURN"
                if prediction == 1
                else "NO CHURN"
            )


        # ====================================================
        # RISK
        # ====================================================

        st.divider()

        st.subheader(
            "📈 Churn Risk Analysis"
        )


        if probability >= 0.75:

            risk = "VERY HIGH"

            st.error(
                "🔴 VERY HIGH RISK"
            )

            recommendation = """
            • Contact the customer immediately.

            • Provide a personalized retention offer.

            • Investigate service problems.

            • Consider loyalty incentives.
            """


        elif probability >= 0.50:

            risk = "HIGH"

            st.warning(
                "🟠 HIGH RISK"
            )

            recommendation = """
            • Send a personalized retention offer.

            • Monitor customer activity.

            • Improve customer support.

            • Consider contract incentives.
            """


        elif probability >= 0.30:

            risk = "MEDIUM"

            st.info(
                "🟡 MEDIUM RISK"
            )

            recommendation = """
            • Monitor customer engagement.

            • Provide proactive support.

            • Track service usage.
            """


        else:

            risk = "LOW"

            st.success(
                "🟢 LOW RISK"
            )

            recommendation = """
            • Continue normal engagement.

            • Maintain service quality.

            • Consider loyalty programs.
            """


        # ====================================================
        # PROGRESS BAR
        # ====================================================

        st.subheader(
            "Churn Probability"
        )

        st.progress(
            probability
        )


        # ====================================================
        # RECOMMENDATION
        # ====================================================

        st.subheader(
            "💡 Recommended Action"
        )

        st.write(
            recommendation
        )


        # ====================================================
        # FINAL SUMMARY
        # ====================================================

        st.divider()

        st.subheader(
            "📋 Prediction Summary"
        )


        summary = pd.DataFrame({

            "Metric": [

                "Prediction",

                "Churn Probability",

                "Risk Level",

                "Tenure",

                "Monthly Charges",

                "Estimated Lifetime Value"
            ],

            "Value": [

                "CHURN"
                if prediction == 1
                else "NO CHURN",

                f"{probability * 100:.2f}%",

                risk,

                f"{tenure} months",

                f"${float(monthly_charges):,.2f}",

                f"${estimated_lifetime_value:,.2f}"
            ]
        })


        # All columns are strings.
        # This is Arrow-safe.

        st.dataframe(

            summary,

            width="stretch",

            hide_index=True
        )


    except Exception as e:

        st.error(
            "❌ Prediction failed."
        )

        st.exception(e)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Customer Churn Prediction | "
    "Python • Pandas • Scikit-learn • Streamlit"
)

