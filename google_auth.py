import requests
from urllib.parse import urlencode
import streamlit as st
import config

class GoogleAuth:
    def __init__(self):
        self.client_id = config.GOOGLE_CLIENT_ID
        self.client_secret = config.GOOGLE_CLIENT_SECRET
        self.redirect_uri = config.GOOGLE_REDIRECT_URI
        
    def get_auth_url(self):
        """Generate Google OAuth authorization URL"""
        params = {
            'client_id': self.client_id,
            'redirect_uri': self.redirect_uri,
            'scope': 'openid email profile',
            'response_type': 'code',
            'access_type': 'offline',
            'prompt': 'consent'
        }
        
        auth_url = f"https://accounts.google.com/o/oauth2/auth?{urlencode(params)}"
        return auth_url
    
    def exchange_code_for_token(self, code):
        """Exchange authorization code for access token"""
        token_url = "https://oauth2.googleapis.com/token"
        
        data = {
            'client_id': self.client_id,
            'client_secret': self.client_secret,
            'code': code,
            'grant_type': 'authorization_code',
            'redirect_uri': self.redirect_uri
        }
        
        response = requests.post(token_url, data=data)
        return response.json()
    
    def get_user_info(self, access_token):
        """Get user information from Google"""
        user_info_url = f"https://www.googleapis.com/oauth2/v2/userinfo?access_token={access_token}"
        response = requests.get(user_info_url)
        return response.json()
    
    def authenticate_user(self, code):
        """Complete authentication flow"""
        try:
            # Exchange code for token
            token_data = self.exchange_code_for_token(code)
            
            if 'access_token' not in token_data:
                return None, f"Token exchange failed: {token_data.get('error', 'Unknown error')}"
            
            # Get user info
            user_info = self.get_user_info(token_data['access_token'])
            
            if 'id' not in user_info:
                return None, f"Failed to get user info: {user_info.get('error', 'Unknown error')}"
            
            return {
                'google_id': user_info['id'],
                'email': user_info['email'],
                'name': user_info['name'],
            }, None
            
        except Exception as e:
            return None, f"Authentication error: {str(e)}"

def init_google_auth():
    """Initialize Google Auth"""
    if not config.GOOGLE_CLIENT_ID:
        st.error("Google OAuth not configured. Set GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET in .env file.")
        return None
    return GoogleAuth()