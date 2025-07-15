import firebase_admin
from firebase_admin import credentials, messaging
from functools import lru_cache

# ✅ Initialize Firebase app once (singleton)
@lru_cache()
def init_firebase():
    cred = credentials.Certificate("app/secrets/firebase-service-account.json")  # path to your JSON file
    firebase_admin.initialize_app(cred)

# ✅ Send push via FCM v1 API
def send_push_notification_v1(token: str, title: str, body: str, screen: str = "HomeScreen"):
    init_firebase()  # ensure Firebase is initialized once

    # Construct message
    message = messaging.Message(
        
        token=token,
        notification=messaging.Notification(
            title=title,
            body=body
        ),
        data={  # Custom payload for routing on tap
            "click_action": "FLUTTER_NOTIFICATION_CLICK",
            "screen": screen
        },
        android=messaging.AndroidConfig(
            priority="high",
            notification=messaging.AndroidNotification(
                sound="default",
                click_action="FLUTTER_NOTIFICATION_CLICK"
            )
        ),
        apns=messaging.APNSConfig(
            headers={"apns-priority": "10"},
            payload=messaging.APNSPayload(
                aps=messaging.Aps(
                    sound="default",
                    category=screen  # optional for iOS
                )
            )
        )
    )

    # Send notification
    response = messaging.send(message)
    print(f"✅ Push notification sent: {response}")
    return {"message_id": response}
