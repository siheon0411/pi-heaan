from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import piheaan as heaan
from piheaan.math import approx
import pandas as pd
import numpy as np
import time

app = FastAPI()
templates = Jinja2Templates(directory="templates")

params = heaan.ParameterPreset.FGb
context = heaan.make_context(params)
heaan.make_bootstrappable(context)

# Load pre-existing key
key_file_path = "./keys"
sk = heaan.SecretKey(context, key_file_path + "/secretkey.bin")
pk = heaan.KeyPack(context, key_file_path + "/")
pk.load_enc_key()
pk.load_mult_key()

eval = heaan.HomEvaluator(context, pk)
dec = heaan.Decryptor(context)
enc = heaan.Encryptor(context)

log_slots = 15
num_slots = 2 ** log_slots

def normalize_data(arr):
    S = sum(arr)
    return [x / S for x in arr]

def step(learning_rate, ctxt_X, ctxt_Y, ctxt_beta, n, log_slots, context, eval):
    ctxt_rot = heaan.Ciphertext(context)
    ctxt_tmp = heaan.Ciphertext(context)

    # Step 1
    ctxt_beta0 = heaan.Ciphertext(context)
    eval.left_rotate(ctxt_beta, 8 * n, ctxt_beta0)
    
    eval.mult(ctxt_beta, ctxt_X, ctxt_tmp)
    for i in range(3):
        eval.left_rotate(ctxt_tmp, n * 2 ** (2 - i), ctxt_rot)
        eval.add(ctxt_tmp, ctxt_rot, ctxt_tmp)
    eval.add(ctxt_tmp, ctxt_beta0, ctxt_tmp)
    
    msg_mask = heaan.Message(log_slots)
    for i in range(n):
        msg_mask[i] = 1
    eval.mult(ctxt_tmp, msg_mask, ctxt_tmp)

    # Step 2
    approx.sigmoid(eval, ctxt_tmp, ctxt_tmp, 8.0)
    eval.bootstrap(ctxt_tmp, ctxt_tmp)
    for i in range(n, num_slots):
        msg_mask[i] = 0.5
    eval.sub(ctxt_tmp, msg_mask, ctxt_tmp)

    # Step 3
    ctxt_d = heaan.Ciphertext(context)
    eval.sub(ctxt_Y, ctxt_tmp, ctxt_d)
    eval.mult(ctxt_d, learning_rate / n, ctxt_d)
    
    eval.right_rotate(ctxt_d, 8 * n, ctxt_tmp)
    for i in range(3):
        eval.right_rotate(ctxt_d, n * 2 ** i, ctxt_rot)
        eval.add(ctxt_d, ctxt_rot, ctxt_d)
    eval.add(ctxt_d, ctxt_tmp, ctxt_d)

    # Step 4
    ctxt_X_j = heaan.Ciphertext(context)
    msg_X0 = heaan.Message(log_slots)
    for i in range(8 * n, 9 * n):
        msg_X0[i] = 1
    eval.add(ctxt_X, msg_X0, ctxt_X_j)
    eval.mult(ctxt_X_j, ctxt_d, ctxt_d)

    # Step 5
    for i in range(9):
        eval.left_rotate(ctxt_d, 2 ** (8 - i), ctxt_rot)
        eval.add(ctxt_d, ctxt_rot, ctxt_d)
    for i in range(9):
        msg_mask[i * n] = 1
    eval.mult(ctxt_d, msg_mask, ctxt_d)

    for i in range(9):
        eval.right_rotate(ctxt_d, 2 ** i, ctxt_rot)
        eval.add(ctxt_d, ctxt_rot, ctxt_d)

    # Step 6
    eval.add(ctxt_beta, ctxt_d, ctxt_d)
    return ctxt_d

def compute_sigmoid(ctxt_X, ctxt_beta, n, log_slots, eval, context, num_slots):
    ctxt_rot = heaan.Ciphertext(context)
    ctxt_tmp = heaan.Ciphertext(context)
    
    ctxt_beta0 = heaan.Ciphertext(context)
    eval.left_rotate(ctxt_beta, 8 * n, ctxt_beta0)
    
    eval.mult(ctxt_beta, ctxt_X, ctxt_tmp)
    for i in range(3):
        eval.left_rotate(ctxt_tmp, n * 2 ** (2 - i), ctxt_rot)
        eval.add(ctxt_tmp, ctxt_rot, ctxt_tmp)
    eval.add(ctxt_tmp, ctxt_beta0, ctxt_tmp)
    
    msg_mask = heaan.Message(log_slots)
    for i in range(n):
        msg_mask[i] = 1
    eval.mult(ctxt_tmp, msg_mask, ctxt_tmp)

    approx.sigmoid(eval, ctxt_tmp, ctxt_tmp, 8.0)
    eval.bootstrap(ctxt_tmp, ctxt_tmp)
    for i in range(n, num_slots):
        msg_mask[i] = 0.5
    eval.sub(ctxt_tmp, msg_mask, ctxt_tmp)
    
    return ctxt_tmp

@app.get("/", response_class=HTMLResponse)
async def get_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request,
                  BMI: float = Form(...),
                  Forehead_circumference: float = Form(...),
                  Neck_circumference: float = Form(...),
                  Armpit_circumference: float = Form(...),
                  Bust: float = Form(...),
                  Rib_cage: float = Form(...),
                  Waist_circumference: float = Form(...),
                  Iliac_circumference: float = Form(...),
                  Femur_circumference: float = Form(...),
                  UrineNighttime_urination: float = Form(...),
                  Personality_Bold_Sensitive: int = Form(...),
                  Personality_Fast_Acting_Slow_Acting: int = Form(...),
                  Personality_Active_Passive: int = Form(...),
                  Personality_Direct_Indirect: int = Form(...),
                  Personality_Fast_to_give_up_Slow_to_give_up: int = Form(...),
                  Personality_Extroverted_Introverted: int = Form(...),
                  Personality_Dynamic_Static: int = Form(...),
                  Personality_Easy_Difficult: int = Form(...),
                  Personality_Masculine_Feminine: int = Form(...),
                  Personality_Impatient_Finds_things_easily: int = Form(...),
                  Personality_Big_Little: int = Form(...),
                  Personality_Extroverted_Subtle: int = Form(...),
                  Personality_Expressive_Expressive: int = Form(...),
                  Personality_Often_excited_Rational: int = Form(...),
                  Personality_Poorly_organized: int = Form(...),
                  Sweating: int = Form(...),
                  SweatMood: int = Form(...),
                  StoolHabits: int = Form(...),
                  Fecal_Bulge: int = Form(...),
                  Feeling_of_urgency_when_defecating: int = Form(...),
                  Feeling_of_loose_stools_after_defecation: int = Form(...)):
    
    start_time = time.time()
    
    input_data = {
        'BMI': BMI,
        'Forehead_circumference': Forehead_circumference,
        'Neck_circumference': Neck_circumference,
        'Armpit_circumference': Armpit_circumference,
        'Bust': Bust,
        'Rib_cage': Rib_cage,
        'Waist_circumference': Waist_circumference,
        'Iliac_circumference': Iliac_circumference,
        'Femur_circumference': Femur_circumference,
        'UrineNighttime_urination': UrineNighttime_urination,
        'Personality_Bold_Sensitive': Personality_Bold_Sensitive,
        'Personality_Fast_Acting_Slow_Acting': Personality_Fast_Acting_Slow_Acting,
        'Personality_Active_Passive': Personality_Active_Passive,
        'Personality_Direct_Indirect': Personality_Direct_Indirect,
        'Personality_Fast_to_give_up_Slow_to_give_up': Personality_Fast_to_give_up_Slow_to_give_up,
        'Personality_Extroverted_Introverted': Personality_Extroverted_Introverted,
        'Personality_Dynamic_Static': Personality_Dynamic_Static,
        'Personality_Easy_Difficult': Personality_Easy_Difficult,
        'Personality_Masculine_Feminine': Personality_Masculine_Feminine,
        'Personality_Impatient_Finds_things_easily': Personality_Impatient_Finds_things_easily,
        'Personality_Big_Little': Personality_Big_Little,
        'Personality_Extroverted_Subtle': Personality_Extroverted_Subtle,
        'Personality_Expressive_Expressive': Personality_Expressive_Expressive,
        'Personality_Often_excited_Rational': Personality_Often_excited_Rational,
        'Personality_Poorly_organized': Personality_Poorly_organized,
        'Sweating': Sweating,
        'SweatMood': SweatMood,
        'StoolHabits': StoolHabits,
        'Fecal_Bulge': Fecal_Bulge,
        'Feeling_of_urgency_when_defecating': Feeling_of_urgency_when_defecating,
        'Feeling_of_loose_stools_after_defecation': Feeling_of_loose_stools_after_defecation
    }
    
    # Prepare the input data for encryption
    X = []
    X.append(normalize_data([input_data['BMI']]))
    X.append(normalize_data([input_data['Forehead_circumference']]))
    X.append(normalize_data([input_data['Neck_circumference']]))
    X.append(normalize_data([input_data['Armpit_circumference']]))
    X.append(normalize_data([input_data['Bust']]))
    X.append(normalize_data([input_data['Rib_cage']]))
    X.append(normalize_data([input_data['Waist_circumference']]))
    X.append(normalize_data([input_data['Iliac_circumference']]))
    X.append(normalize_data([input_data['Femur_circumference']]))
    X.append([input_data['Personality_Bold_Sensitive']])
    X.append([input_data['Personality_Fast_Acting_Slow_Acting']])
    X.append([input_data['Personality_Active_Passive']])
    X.append([input_data['Personality_Direct_Indirect']])
    X.append([input_data['Personality_Fast_to_give_up_Slow_to_give_up']])
    X.append([input_data['Personality_Extroverted_Introverted']])
    X.append([input_data['Personality_Dynamic_Static']])
    X.append([input_data['Personality_Easy_Difficult']])
    X.append([input_data['Personality_Masculine_Feminine']])
    X.append([input_data['Personality_Impatient_Finds_things_easily']])
    X.append([input_data['Personality_Big_Little']])
    X.append([input_data['Personality_Extroverted_Subtle']])
    X.append([input_data['Personality_Expressive_Expressive']])
    X.append([input_data['Personality_Often_excited_Rational']])
    X.append([input_data['Personality_Poorly_organized']])
    X.append([input_data['Sweating']])
    X.append([input_data['SweatMood']])
    X.append([input_data['StoolHabits']])
    X.append([input_data['Fecal_Bulge']])
    X.append([input_data['Feeling_of_urgency_when_defecating']])
    X.append([input_data['Feeling_of_loose_stools_after_defecation']])
    X.append(normalize_data([input_data['UrineNighttime_urination']]))

    # Encrypt the input data
    msg_X = heaan.Message(log_slots)
    ctxt_X = heaan.Ciphertext(context)
    for i in range(31):
        msg_X[i] = X[i][0]
    enc.encrypt(msg_X, pk, ctxt_X)

    # Encrypt the initial beta
    beta = 2 * np.random.rand(32) - 1
    msg_beta = heaan.Message(log_slots)
    ctxt_beta = heaan.Ciphertext(context)
    for i in range(31):
        msg_beta[i] = beta[i + 1]
    msg_beta[31] = beta[0]
    enc.encrypt(msg_beta, pk, ctxt_beta)

    # Run the homomorphic logistic regression
    num_steps = 100
    ctxt_next = ctxt_beta
    for i in range(num_steps):
        ctxt_next = step(0.2, ctxt_X, ctxt_X, ctxt_next, 1, log_slots, context, eval)

    # Decrypt the result
    res_beta = heaan.Message(log_slots)
    dec.decrypt(ctxt_next, sk, res_beta)

    # Predict the output using the computed beta
    ctxt_infer = compute_sigmoid(ctxt_X, ctxt_next, 1, log_slots, eval, context, num_slots)
    res = heaan.Message(log_slots)
    dec.decrypt(ctxt_infer, sk, res)

    # Calculate the accuracy (dummy value for demonstration)
    accuracy = 0.95  # Placeholder for actual accuracy calculation
    execution_time = time.time() - start_time

    return templates.TemplateResponse("index.html", {
        "request": request,
        "prediction": res[0].real,
        "accuracy": accuracy,
        "execution_time": execution_time
    })

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=8000, log_level="info")