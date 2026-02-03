from twilio.rest import Client

# ================================
# TWILIO CONFIGURATION
# ================================
ACCOUNT_SID = ""
AUTH_TOKEN = ""

TWILIO_NUMBER = "+19894454386"   # Twilio number (NO SPACES)
TO_NUMBER="+919790235865"
# Example TO number format: +91XXXXXXXXXX

# ================================
# SEND SMS FUNCTION
# ================================
def send_sms(to_number, temp, prediction, level, suggestions):
    try:
        client = Client(ACCOUNT_SID, AUTH_TOKEN)

        # SAFE, SHORT, ASCII-ONLY MESSAGE
        message_body = (
            "Energy Alert\n"
            f"Temp: {temp} C\n"
            f"Usage: {prediction} kWh\n"
            f"Level: {level}\n"
            "Tips:\n"
            f"1. {suggestions[0]}\n"
            f"2. {suggestions[1]}"
        )

        message = client.messages.create(
            body=message_body,
            from_=TWILIO_NUMBER,
            to=to_number
        )

        print("SMS sent successfully. SID:", message.sid)
        return True

    except Exception as e:
        print("SMS failed:", e)
        return False
