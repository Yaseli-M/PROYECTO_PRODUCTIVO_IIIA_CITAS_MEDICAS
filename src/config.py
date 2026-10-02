import os
from dotenv import load_dotenv
from supabase import create_client, Client

# Cargar las variables desde el archivo .env
load_dotenv()

url: str = os.getenv("SUPABASE_URL")
key: str = os.getenv("SUPABASE_KEY")

if not url or not key:
    raise ValueError(" No se encontraron SUPABASE_URL o SUPABASE_KEY en el archivo .env")

# Cliente de conexión a Supabase
supabase: Client = create_client(url, key)