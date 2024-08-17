import google.generativeai as genAI
import pandas as pd

model = genAI.GenerativeModel("gemini-pro")


def make_prompt(p: pd.DataFrame) -> str:
    
    return f"""
            Emosi seseorang adalah {p.loc[0, "Emotional"]} dengan probabilitas {p.loc[0, "Emotional"]} {p.loc[0, "Probability"]}%, {p.loc[1, "Emotional"]} {p.loc[1, "Probability"]}%, {p.loc[2, "Emotional"]} {p.loc[2, "Probability"]}%, {p.loc[3, "Emotional"]} {p.loc[3, "Probability"]}%, {p.loc[4, "Emotional"]} {p.loc[4, "Probability"]}%, dan {p.loc[5, "Emotional"]} {p.loc[5, "Probability"]}%.\nBuatlah kesimpulan dari hasil prediksi itu dan berikan tips atau saran dari apa yang dia rasakan dan buatkan quotes dari tokoh terkenal yang berhubungan dengan apa yang dia rasakan. catatan: abaikan emosi yang persentasenya dibawah 10%. Gunakan bahasa Indonesia, ganti kata 'seseorang' menjadi 'kamu', jangan gunakan kata 'kesimpulan', jangan gunakan kata 'Qoutes: dan tambahkan simbol '>' pada bagian quotes untuk membuat Continuing the Blockquote pada markdown'
    """
    
def get_summary(prompt: str) -> str:
    response = model.generate_content(prompt)
    return response.text