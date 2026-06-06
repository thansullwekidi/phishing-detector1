"""
Train Model Phishing Detection
Melatih model AI untuk deteksi website phishing menggunakan Random Forest Classifier
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import os

# ==========================================
# 1. GENERATE SIMULASI DATASET PHISHING
# ==========================================

def generate_phishing_dataset(n_samples=500):
    """
    Generate simulasi dataset phishing dengan fitur-fitur:
    - panjang_url: panjang URL dalam karakter
    - ada_karakter_at: ada tidaknya karakter '@' di URL (1=ada, 0=tidak)
    - jumlah_titik: jumlah karakter titik '.' di URL
    """
    
    print("🔄 Generating simulasi dataset phishing...")
    
    data = {
        'panjang_url': [],
        'ada_karakter_at': [],
        'jumlah_titik': [],
        'label': []  # 0 = Aman, 1 = Phishing
    }
    
    np.random.seed(42)
    
    # Generate URL Aman (label = 0)
    for _ in range(n_samples // 2):
        panjang = np.random.randint(20, 60)  # URL normal biasanya lebih pendek
        ada_at = 0  # URL aman tidak punya @
        titik = np.random.randint(1, 4)  # 1-3 titik untuk domain aman
        
        data['panjang_url'].append(panjang)
        data['ada_karakter_at'].append(ada_at)
        data['jumlah_titik'].append(titik)
        data['label'].append(0)
    
    # Generate URL Phishing (label = 1)
    for _ in range(n_samples // 2):
        panjang = np.random.randint(60, 150)  # URL phishing biasanya lebih panjang
        ada_at = np.random.choice([0, 1], p=[0.3, 0.7])  # 70% phishing punya @
        titik = np.random.randint(2, 8)  # Lebih banyak titik untuk domain phishing
        
        data['panjang_url'].append(panjang)
        data['ada_karakter_at'].append(ada_at)
        data['jumlah_titik'].append(titik)
        data['label'].append(1)
    
    df = pd.DataFrame(data)
    print(f"✅ Dataset berhasil dibuat dengan {len(df)} sampel")
    print(f"\n📊 Distribusi data:")
    print(df['label'].value_counts())
    
    return df


# ==========================================
# 2. SIMPAN DATASET KE FILE CSV
# ==========================================

def save_dataset(df, filename='dataset_phishing.csv'):
    """Simpan dataset ke file CSV"""
    df.to_csv(filename, index=False)
    print(f"\n💾 Dataset berhasil disimpan ke '{filename}'")


# ==========================================
# 3. LATIH MODEL RANDOM FOREST
# ==========================================

def train_model(df):
    """
    Latih model Random Forest Classifier
    """
    print("\n🤖 Memulai training model Random Forest...")
    
    # Pisahkan fitur (X) dan label (y)
    X = df[['panjang_url', 'ada_karakter_at', 'jumlah_titik']]
    y = df['label']
    
    # Split data: 80% training, 20% testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Buat dan latih model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    print("✅ Model training selesai!")
    
    # Evaluasi model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n📈 Model Performance:")
    print(f"Akurasi: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"\n📊 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Aman', 'Phishing']))
    print(f"\n🎯 Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    return model


# ==========================================
# 4. SIMPAN MODEL KE FILE PICKLE
# ==========================================

def save_model(model, filename='model_phishing.pkl'):
    """Simpan model ke file binary pickle"""
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    print(f"\n💾 Model berhasil disimpan ke '{filename}'")


# ==========================================
# MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    print("=" * 50)
    print("🛡️  TRAINING MODEL DETEKSI PHISHING")
    print("=" * 50)
    
    # Step 1: Generate dataset
    df = generate_phishing_dataset(n_samples=500)
    
    # Step 2: Simpan dataset
    save_dataset(df, 'dataset_phishing.csv')
    
    # Step 3: Latih model
    model = train_model(df)
    
    # Step 4: Simpan model
    save_model(model, 'model_phishing.pkl')
    
    print("\n" + "=" * 50)
    print("✅ SEMUA PROSES SELESAI!")
    print("=" * 50)
    print("File yang dibuat:")
    print("  1. dataset_phishing.csv - Dataset untuk training")
    print("  2. model_phishing.pkl - Model yang sudah terlatih")
    print("\nSelanjutnya, jalankan 'app.py' untuk membuka aplikasi Streamlit")
    print("=" * 50)
