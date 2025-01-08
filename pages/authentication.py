import os
import streamlit as st
from dotenv import load_dotenv  # Import dotenv

from supabase import create_client, Client

st.title("Authentication Page")
load_dotenv()

# Retrieve Supabase credentials from environment variables
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_ANON_KEY = os.environ.get("SUPABASE_ANON_KEY")

# Initialize the Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

# Display login/registration form
mode = st.selectbox("Select Mode", ["Login", "Register"])
email = st.text_input("Email")
password = st.text_input("Password", type="password")

if st.button("Submit"):
    if mode == "Login":
        # Attempt to sign the user in
        auth_response = supabase.auth.sign_in_with_password({"email": email, "password": password})
        if "error" in auth_response and auth_response["error"]:
            st.error("Login failed. Check your credentials.")
        else:
            st.success("Logged in successfully!")
            st.session_state["logged_in"] = True
    else:
        # Attempt to sign the user up
        auth_response = supabase.auth.sign_up({"email": email, "password": password})

        if "error" in auth_response and auth_response["error"]:
            st.error("Registration failed. Try a different email or password.")
        else:
            st.success("Registered successfully! You can now switch to Login mode.")

# Provide a way to log out
if st.session_state.get("logged_in"):
    if st.button("Logout"):
        supabase.auth.sign_out()
        st.session_state["logged_in"] = False
        st.success("You have logged out.")
