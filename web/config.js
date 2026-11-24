// Where the BechoAI API lives: same server when run locally, the Render API everywhere else (Vercel, Render).
window.BECHOAI_API = ["localhost", "127.0.0.1"].includes(location.hostname) ? "" : "https://bechoai-api.onrender.com";
