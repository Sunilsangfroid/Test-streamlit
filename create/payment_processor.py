import streamlit as st
import pandas as pd
from datetime import datetime
import random

def display_payment_methods(key_suffix="default"):
    """Display payment method selection with a unique key"""
    payment_method = st.selectbox(
        "Select Payment Method",
        ["Credit Card", "Debit Card", "UPI", "Net Banking", "Wallet"],
        key=f"payment_method_{key_suffix}"  # Add unique key
    )
    
    # Display different payment forms based on selection
    if payment_method == "Credit Card":
        st.text_input("Card Number", placeholder="XXXX XXXX XXXX XXXX", key=f"cc_num_{key_suffix}")
        col1, col2 = st.columns(2)
        with col1:
            st.text_input("Expiry Date", placeholder="MM/YY", key=f"cc_exp_{key_suffix}")
        with col2:
            st.text_input("CVV", placeholder="XXX", type="password", key=f"cc_cvv_{key_suffix}")
        st.text_input("Name on Card", key=f"cc_name_{key_suffix}")
    
    elif payment_method == "Debit Card":
        st.text_input("Card Number", placeholder="XXXX XXXX XXXX XXXX", key=f"dc_num_{key_suffix}")
        col1, col2 = st.columns(2)
        with col1:
            st.text_input("Expiry Date", placeholder="MM/YY", key=f"dc_exp_{key_suffix}")
        with col2:
            st.text_input("CVV", placeholder="XXX", type="password", key=f"dc_cvv_{key_suffix}")
        st.text_input("Name on Card", key=f"dc_name_{key_suffix}")
    
    elif payment_method == "UPI":
        st.text_input("UPI ID", placeholder="username@upi", key=f"upi_id_{key_suffix}")
    
    elif payment_method == "Net Banking":
        st.selectbox(
            "Select Bank",
            ["SBI", "HDFC", "ICICI", "Axis", "PNB", "Other"],
            key=f"netbank_select_{key_suffix}"
        )
        st.text_input("User ID", key=f"netbank_id_{key_suffix}")
        st.text_input("Password", type="password", key=f"netbank_pass_{key_suffix}")
    
    elif payment_method == "Wallet":
        st.selectbox(
            "Select Wallet",
            ["Paytm", "PhonePe", "Google Pay", "Amazon Pay"],
            key=f"wallet_select_{key_suffix}"
        )
        st.text_input("Mobile Number", key=f"wallet_mobile_{key_suffix}")
    
    return payment_method

def process_payment(amount, payment_method, key_suffix="default"):
    """Process a payment transaction"""
    transaction_id = f"TX-{random.randint(1000000, 9999999)}"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    return {
        "transaction_id": transaction_id,
        "amount": amount,
        "payment_method": payment_method,
        "status": "Successful",
        "timestamp": timestamp
    }

def display_payment_summary():
    """Display a summary of all payments made"""
    if "payments" in st.session_state and st.session_state.payments:
        st.subheader("Payment Summary")
        
        payments = st.session_state.payments
        df = pd.DataFrame(payments)
        
        # Calculate total
        total = sum(payment["amount"] for payment in payments)
        
        # Display table
        st.dataframe(
            df[["transaction_id", "booking_type", "amount", "payment_method", "status"]],
            hide_index=True,
            use_container_width=True
        )
        
        # Display total
        st.info(f"**Total Amount Paid:** ₹{total:,.2f}")
        
        # Receipt options
        st.download_button(
            "Download Payment Receipt",
            "Payment receipt content would go here",
            file_name="payment_receipt.pdf",
            mime="application/pdf",
        )
    else:
        st.info("No payments have been made yet.")