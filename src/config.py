import os
from dotenv import load_dotenv
from supabase import create_client, Client

# Cargar las variables definidas en el archivo .env
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError(
        " No se encontraron 'SUPABASE_URL' o 'SUPABASE_KEY' en el archivo .env. "
        "Verifica que el archivo .env exista en la raíz del proyecto."
    )

# Cliente global de Supabase para importar en los scripts
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)