from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import pandas as pd
import numpy as np
import piheaan as heaan
from piheaan.math import approx
import joblib
import time

app = FastAPI()

# static file directory setting
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def read_root():
    return {"message": "Welcome to the HEAAN API"}

# Favicon directory setting
@app.get("/favicon.ico")
def favicon():
    return "Favicon not found", 404

# homomorphic encryption setting
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
num_slots = 2**log_slots

# function to normalize data
def normalize_data(arr):
    S = sum(arr)
    return [x / S for x in arr]

# step function to train logistic regression model
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

# calculating through sigmoid function
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

# load train dataset and do preprocessing
csv_train = pd.read_csv('./data/train_df.csv')
df_train = pd.DataFrame(csv_train)

train_n = df_train.shape[0]
X_train = [normalize_data(df_train[col].values) if col not in [
            'personality_bs', 'personality_fasa', 'personality_ap', 
            'personality_di', 'personality_fgsg', 'personality_ei', 
            'personality_ds', 'personality_ed', 'personality_mf', 
            'personality_ifte', 'personality_bl', 'personality_es', 
            'personality_ee', 'personality_oxr', 'personality_po', 
            'sweating', 'sweatmood', 'stoolhabits', 'fecal_bulge', 
            'fouw_defecating', 'folsa_defecation'] else list(df_train[col].values) 
        for col in df_train.columns if col != 'km_diagnosis']
Y_train = list(df_train['km_diagnosis'].values)

msg_X_train = heaan.Message(log_slots)
ctxt_X_train = heaan.Ciphertext(context)
for i in range(31):
    for j in range(train_n):
        msg_X_train[train_n * i + j] = X_train[i][j]
enc.encrypt(msg_X_train, pk, ctxt_X_train)

msg_Y_train = heaan.Message(log_slots)
ctxt_Y_train = heaan.Ciphertext(context)
for j in range(train_n):
    msg_Y_train[j] = Y_train[j]
enc.encrypt(msg_Y_train, pk, ctxt_Y_train)

beta = 2 * np.random.rand(32) - 1
msg_beta = heaan.Message(log_slots)
ctxt_beta = heaan.Ciphertext(context)

for i in range(31):
    for j in range(train_n):
        msg_beta[train_n * i + j] = beta[i + 1]
for j in range(train_n):
    msg_beta[31 * train_n + j] = beta[0]

enc.encrypt(msg_beta, pk, ctxt_beta)
learning_rate = 0.2
num_steps = 100

ctxt_next = heaan.Ciphertext(context)
eval.add(ctxt_beta, 0, ctxt_next)

start_time = time.time()
for i in range(num_steps):
    print("=== Step", i, "===")
    ctxt_next = step(0.2, ctxt_X_train, ctxt_Y_train, ctxt_next, train_n, log_slots, context, eval)
execution_time = time.time() - start_time
print('model training execution time: ', execution_time)

# load test dataset
csv_test = pd.read_csv('./data/test_df.csv')
df_test = pd.DataFrame(csv_test)
test_n = df_test.shape[0]

X_test = [normalize_data(df_test[col].values) if col not in [
            'personality_bs', 'personality_fasa', 'personality_ap', 
            'personality_di', 'personality_fgsg', 'personality_ei', 
            'personality_ds', 'personality_ed', 'personality_mf', 
            'personality_ifte', 'personality_bl', 'personality_es', 
            'personality_ee', 'personality_oxr', 'personality_po', 
            'sweating', 'sweatmood', 'stoolhabits', 'fecal_bulge', 
            'fouw_defecating', 'folsa_defecation'] else list(df_test[col].values) 
        for col in df_test.columns if col != 'km_diagnosis']
Y_test = list(df_test['km_diagnosis'].values)

msg_X_test = heaan.Message(log_slots)
ctxt_X_test = heaan.Ciphertext(context)
for i in range(31):
    for j in range(test_n):
        msg_X_test[test_n * i + j] = X_test[i][j]
enc.encrypt(msg_X_test, pk, ctxt_X_test)

# calculate model accuracy
ctxt_infer = compute_sigmoid(ctxt_X_test, ctxt_next, test_n, log_slots, eval, context, num_slots)

res = heaan.Message(log_slots)
dec.decrypt(ctxt_infer, sk, res)
cnt = 0
for i in range(test_n):
    if res[i].real >= 0.6:
        if Y_test[i] == 1:
            cnt += 1
    else:
        if Y_test[i] == 0:
            cnt += 1
print("Accuracy : ", cnt / test_n)

# set the datamodel for the input
class Data(BaseModel):
    bmi: float
    forehead_circumference: float
    neck_circumference: float
    armpit_circumference: float
    bust: float
    rib_cage: float
    waist_circumference: float
    iliac_circumference: float
    femur_circumference: float
    personality_bs: int
    personality_fasa: int
    personality_ap: int
    personality_di: int
    personality_fgsg: int
    personality_ei: int
    personality_ds: int
    personality_ed: int
    personality_mf: int
    personality_ifte: int
    personality_bl: int
    personality_es: int
    personality_ee: int
    personality_oxr: int
    personality_po: int
    sweating: int
    sweatmood: int
    stoolhabits: int
    fecal_bulge: int
    fouw_defecating: int
    folsa_defecation: int
    urinenighttime_urination: int

@app.post("/predict")
# get input and predict class through model
def predict(data: Data):
    try:
        start_time = time.time()

        # convert data to DataFrame
        input_data = pd.DataFrame([data.dict().values()], columns=data.dict().keys())
        print(input_data)

        # set columns to normalize
        columns_to_normalize = [
            'bmi', 'forehead_circumference', 'neck_circumference', 'armpit_circumference',
            'bust', 'rib_cage', 'waist_circumference', 'iliac_circumference', 'femur_circumference',
            'urinenighttime_urination'
        ]

        # execute normalizing
        for col in columns_to_normalize:
            input_data[col] = normalize_data(input_data[col].values)

        # data encryption
        msg_X = heaan.Message(log_slots)
        ctxt_X = heaan.Ciphertext(context)
        for i, col in enumerate(input_data.columns):
            for j in range(len(input_data)):
                msg_X[len(input_data) * i + j] = input_data[col].iloc[j]
        enc.encrypt(msg_X, pk, ctxt_X)

       # model prediction
        ctxt_result = compute_sigmoid(ctxt_X, ctxt_next, 1, log_slots, eval, context, num_slots)

        # result decryption
        res = heaan.Message(log_slots)
        dec.decrypt(ctxt_result, sk, res)

        # result of prediction
        prediction = 1 if res[0].real >= 0.6 else 0
        
        # finish calculating execution time
        execution_time = time.time() - start_time

        return {
            "prediction": prediction,
            "execution_time": execution_time
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
