#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clasificador de noticias (Deporte / Política / Tecnología) - from scratch

- Sin sklearn/xgboost/statsmodels. Solo numpy, re, tkinter (pandas opcional para CSV)
- Dataset sintético ES (títulos/leads) y guardado a CSV opcional
- Pipeline: normalización → tokenización → stopwords → TF-IDF (manual)
- Modelo: Regresión Logística One-vs-Rest (3 clases) con descenso de gradiente + L2
- Métricas: accuracy, F1 macro y por clase + matriz de confusión 3×3
- GUI: cuadro de texto (máx. 100 caracteres) → predicción y “probabilidades” OvR

Ejecuta:
    python3 news_classifier_from_scratch.py
"""

import re
import os
import math
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox


try:
    import pandas as pd
    HAS_PANDAS = True
except Exception:
    HAS_PANDAS = False

# ---------------------------
# Configuración
# ---------------------------
RNG_SEED = 42
np.random.seed(RNG_SEED)

MAX_LEN = 100       # límite de caracteres del título/lead de entrada
TOP_VOCAB = 2000    # tamaño máximo del vocabulario TF-IDF
MIN_DF = 1          # doc freq mínima para incluir un término

CLASSES = ["deporte", "politica", "tecnologia"]   # orden fijo de etiquetas
CLASS_TO_ID = {c: i for i, c in enumerate(CLASSES)}

# ---------------------------
# Dataset sintético (ES)
# ---------------------------
def build_synthetic_corpus():
    """
    Devuelve lista de (texto, etiqueta_id) para 3 clases:
      0=deporte, 1=politica, 2=tecnologia
    Incluye ejemplos base + variaciones simples.
    """
    sport = [
        "Messi marca un gol y gana el partido",
        "El Real Madrid gana la final de la copa",
        "La selección avanza a semifinales",
        "El delantero anota un hat trick",
        "El Barcelona ficha a un nuevo defensa",
        "Partido de liga termina en empate",
        "El técnico anuncia la convocatoria",
        "El árbitro expulsa a dos jugadores",
        "Final de la Champions se jugará el sábado",
        "Golazo en el tiempo de descuento"
    ]
    politics = [
        "El congreso aprueba la nueva ley",
        "Elecciones presidenciales este domingo",
        "El presidente anuncia reforma fiscal",
        "Debate político sobre seguridad y salud",
        "La oposición rechaza el presupuesto",
        "El senado discute cambios constitucionales",
        "Ministro presenta informe de gobierno",
        "Protestas frente al parlamento",
        "Nuevo gabinete toma posesión",
        "Tratado internacional entra en vigor"
    ]
    tech = [
        "Nuevo procesador de Intel mejora el rendimiento",
        "Tesla anuncia coche eléctrico actualizado",
        "Lanzan actualización de software para móviles",
        "Apple presenta nuevo modelo de iPhone",
        "Empresa publica avance en inteligencia artificial",
        "Se descubre vulnerabilidad en sistema operativo",
        "Se lanza versión estable de Linux",
        "Anuncian batería de mayor duración",
        "Google revela mejoras en su buscador",
        "Presentan robot doméstico con visión artificial"
    ]

    # pequeñas variaciones (añadir colas y sinónimos simples)
    tails = [" hoy", " este año", " en conferencia", " oficial", " de última hora", " con récord", " en rueda de prensa"]
    def jitter(msgs):
        extra = []
        for m in msgs:
            for t in np.random.choice(tails, size=2, replace=False):
                extra.append((m + t).strip())
        return msgs + extra

    sport2 = jitter(sport)
    politics2 = jitter(politics)
    tech2 = jitter(tech)

    data = [(s, CLASS_TO_ID["deporte"])   for s in sport2] + \
           [(p, CLASS_TO_ID["politica"])  for p in politics2] + \
           [(t, CLASS_TO_ID["tecnologia"]) for t in tech2]

    np.random.shuffle(data)
    return data  # ~90 ejemplos

def save_csv(data, path="news_corpus_es.csv"):
    if not HAS_PANDAS:
        return
    df = pd.DataFrame([(t, CLASSES[y]) for t, y in data], columns=["texto", "clase"])
    df.to_csv(path, index=False, encoding="utf-8")
    print(f"CSV guardado en: {os.path.abspath(path)}")

# ---------------------------
# Preprocesamiento
# ---------------------------
STOPWORDS_ES = set("""
a al algo algunas algunos ante antes como con contra cual cuales cuando de del desde donde dos el ella ellas ellos en entre era eres es esa ese esto estos estoy fue fui fuimos fueron ha han
hasta hay la las le les lo los mas me mi mis mucha mucho muy nada ni no nosotros nosotras para pero poco por porque que quien quienes se sin sobre su sus te tengo tiene tienen tuvo tu tus un una uno unos y ya
""".split())

def normalize_text(s):
    s = s.lower().strip()[:MAX_LEN]
    s = re.sub(r"https?://\S+|www\.\S+", " url ", s)
    s = re.sub(r"[0-9]+", " num ", s)
    s = re.sub(r"[^\wáéíóúñü]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def tokenize(s):
    return [t for t in s.split() if t and t not in STOPWORDS_ES]

# ---------------------------
# TF-IDF (manual)
# ---------------------------
class TfidfVectorizerManual:
    def __init__(self, top_k=TOP_VOCAB, min_df=MIN_DF):
        self.top_k = top_k
        self.min_df = min_df
        self.vocab_ = {}
        self.idf_ = None

    def fit(self, docs_tokens):
        df = {}
        for toks in docs_tokens:
            for w in set(toks):
                df[w] = df.get(w, 0) + 1
        df = {w: d for w, d in df.items() if d >= self.min_df}
        ordered = sorted(df.items(), key=lambda x: (-x[1], x[0]))[:self.top_k]
        self.vocab_ = {w: i for i, (w, _) in enumerate(ordered)}
        N = len(docs_tokens)
        self.idf_ = np.zeros(len(self.vocab_), dtype=float)
        for w, i in self.vocab_.items():
            self.idf_[i] = math.log((N + 1) / (df[w] + 1)) + 1.0
        return self

    def transform(self, docs_tokens):
        V = len(self.vocab_)
        X = np.zeros((len(docs_tokens), V), dtype=float)
        for r, toks in enumerate(docs_tokens):
            if not toks:
                continue
            tf = {}
            for w in toks:
                j = self.vocab_.get(w, -1)
                if j >= 0:
                    tf[j] = tf.get(j, 0.0) + 1.0
            if not tf:
                continue
            max_tf = max(tf.values())
            for j, c in tf.items():
                X[r, j] = (c / max_tf) * self.idf_[j]
        # normalización L2 por fila
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return X / norms

    def fit_transform(self, docs_tokens):
        return self.fit(docs_tokens).transform(docs_tokens)

# ---------------------------
# Split train/test (manual)
# ---------------------------
def train_test_split_lists(token_lists, y, test_size=0.2, seed=RNG_SEED):
    n = len(y)
    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    n_test = int(np.ceil(test_size * n))
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]
    train_tokens = [token_lists[i] for i in train_idx]
    test_tokens  = [token_lists[i] for i in test_idx]
    return np.array(train_tokens, dtype=object), np.array(test_tokens, dtype=object), y[train_idx], y[test_idx]

# ---------------------------
# Regresión Logística OvR
# ---------------------------
class LogisticRegressionBinaryGD:
    """Clasificador binario (para cada clase en OvR)."""
    def __init__(self, lr=0.3, n_epochs=1500, l2=0.001, verbose=False):
        self.lr = lr
        self.n_epochs = n_epochs
        self.l2 = l2
        self.verbose = verbose
        self.w = None  # incluye bias

    @staticmethod
    def _sigmoid(z):
        out = np.empty_like(z, dtype=float)
        m = (z >= 0)
        out[m] = 1.0 / (1.0 + np.exp(-z[m]))
        ez = np.exp(z[~m])
        out[~m] = ez / (1.0 + ez)
        return out

    @staticmethod
    def _add_bias(X):
        return np.hstack([X, np.ones((X.shape[0], 1))])

    def fit(self, X, y):
        Xb = self._add_bias(X)
        n, d = Xb.shape
        self.w = np.zeros(d)
        for ep in range(self.n_epochs):
            p = self._sigmoid(Xb @ self.w)
            err = p - y
            grad = (Xb.T @ err) / n
            if self.l2 > 0.0:
                reg = 2.0 * self.l2 * self.w
                reg[-1] = 0.0
                grad += reg
            self.w -= self.lr * grad
            if self.verbose and ep % max(1, self.n_epochs // 10) == 0:
                print(f"[{ep:4d}] loss={self.loss(X, y):.4f}")
        return self

    def predict_proba(self, X):
        Xb = self._add_bias(X)
        return self._sigmoid(Xb @ self.w)

    def loss(self, X, y):
        Xb = self._add_bias(X)
        p = self._sigmoid(Xb @ self.w)
        eps = 1e-12
        bce = -np.mean(y*np.log(p+eps) + (1-y)*np.log(1-p+eps))
        if self.l2 > 0.0:
            bce += self.l2 * np.sum(self.w[:-1]**2)
        return bce

class LogisticRegressionOVR:
    """One-vs-Rest para K clases usando clasificadores binarios."""
    def __init__(self, n_classes, lr=0.3, n_epochs=1500, l2=0.001, verbose=False):
        self.n_classes = n_classes
        self.models = [LogisticRegressionBinaryGD(lr, n_epochs, l2, verbose) for _ in range(n_classes)]

    def fit(self, X, y):
        # para cada clase k, entrenar y_k = (y==k)
        for k in range(self.n_classes):
            yk = (y == k).astype(int)
            self.models[k].fit(X, yk)
        return self

    def predict_proba(self, X):
        # retornamos las probabilidades OvR por clase (no normalizadas a 1)
        P = np.column_stack([m.predict_proba(X) for m in self.models])  # (n, K)
        # opcional: normalizar fila a simplex (softmax-like) para GUI
        row_sums = P.sum(axis=1, keepdims=True) + 1e-12
        return P / row_sums

    def predict(self, X):
        P = self.predict_proba(X)
        return np.argmax(P, axis=1)

# ---------------------------
# Métricas multiclase
# ---------------------------
def confusion_matrix_3x3(y_true, y_pred, n_classes=3):
    M = np.zeros((n_classes, n_classes), dtype=int)
    for yt, yp in zip(y_true, y_pred):
        M[yt, yp] += 1
    return M

def per_class_precision_recall_f1(cm):
    # cm[i,i] = TP_i; fila i = verdaderos i; columna i = predichos i
    K = cm.shape[0]
    prec, rec, f1 = np.zeros(K), np.zeros(K), np.zeros(K)
    for i in range(K):
        TP = cm[i, i]
        FP = cm[:, i].sum() - TP
        FN = cm[i, :].sum() - TP
        prec[i] = TP / max(TP + FP, 1e-12)
        rec[i]  = TP / max(TP + FN, 1e-12)
        f1[i]   = 2 * prec[i] * rec[i] / max(prec[i] + rec[i], 1e-12)
    return prec, rec, f1

# ---------------------------
# Pipeline de entrenamiento
# ---------------------------
def prepare_and_train():
    data = build_synthetic_corpus()
    if HAS_PANDAS:
        save_csv(data)

    texts = [normalize_text(t) for t, _ in data]
    tokens = [tokenize(t) for t in texts]
    y = np.array([lab for _, lab in data], dtype=int)

    X_tr_tok, X_te_tok, y_tr, y_te = train_test_split_lists(tokens, y, test_size=0.25)

    vec = TfidfVectorizerManual(top_k=TOP_VOCAB, min_df=MIN_DF)
    X_tr = vec.fit_transform(X_tr_tok)
    X_te = vec.transform(X_te_tok)

    model = LogisticRegressionOVR(n_classes=len(CLASSES), lr=0.4, n_epochs=2000, l2=0.0008, verbose=False)
    model.fit(X_tr, y_tr)

    y_hat = model.predict(X_te)
    cm = confusion_matrix_3x3(y_te, y_hat, n_classes=len(CLASSES))
    acc = (y_te == y_hat).mean()
    prec, rec, f1 = per_class_precision_recall_f1(cm)
    f1_macro = f1.mean()

    print("\n=== Evaluación (Test) - Clasificador de Noticias ===")
    print(f"accuracy: {acc:.4f} | f1_macro: {f1_macro:.4f}")
    for i, c in enumerate(CLASSES):
        print(f"  - {c:11s} -> precision: {prec[i]:.4f} | recall: {rec[i]:.4f} | f1: {f1[i]:.4f}")
    print("\nMatriz de confusión (filas=verdadero, cols=predicho):")
    print(cm)

    return model, vec

# ---------------------------
# GUI
# ---------------------------
class NewsGUI:
    def __init__(self, model, vectorizer):
        self.model = model
        self.vectorizer = vectorizer

        self.root = tk.Tk()
        self.root.title("Clasificador de Noticias (ES) - From Scratch")

        frm = ttk.Frame(self.root, padding=10)
        frm.grid(row=0, column=0, sticky="nsew")

        ttk.Label(frm, text=f"Título/lead (máx. {MAX_LEN} caracteres):").grid(row=0, column=0, sticky="w")
        self.text = tk.Text(frm, width=70, height=4, wrap="word")
        self.text.grid(row=1, column=0, pady=6)

        ttk.Button(frm, text="Predecir", command=self.on_predict).grid(row=2, column=0, pady=8, sticky="w")

        self.pred_var = tk.StringVar(value="Predicción: -")
        self.prob_var = tk.StringVar(value="Probabilidades (OvR): -")
        ttk.Label(frm, textvariable=self.pred_var, font=("TkDefaultFont", 11, "bold")).grid(row=3, column=0, sticky="w")
        ttk.Label(frm, textvariable=self.prob_var).grid(row=4, column=0, sticky="w")

        hint = "Clases: deporte / política / tecnología. El modelo usa TF-IDF y logística OvR."
        ttk.Label(frm, text=hint).grid(row=5, column=0, sticky="w", pady=(6,0))

    def on_predict(self):
        try:
            s = self.text.get("1.0", "end").strip()[:MAX_LEN]
            toks = tokenize(normalize_text(s))
            X = self.vectorizer.transform([toks])
            probs = self.model.predict_proba(X)[0]  # (K,)
            k = int(np.argmax(probs))
            cls = CLASSES[k]
            probs_str = ", ".join([f"{CLASSES[i]}={probs[i]:.3f}" for i in range(len(CLASSES))])
            self.pred_var.set(f"Predicción: {cls}")
            self.prob_var.set(f"Probabilidades (OvR): {probs_str}")
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo predecir:\n{e}")

    def run(self):
        self.root.mainloop()

# ---------------------------
# Main
# ---------------------------
def main():
    model, vec = prepare_and_train()
    gui = NewsGUI(model, vec)
    gui.run()

if __name__ == "__main__":
    main()
