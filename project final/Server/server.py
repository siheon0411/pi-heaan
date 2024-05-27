from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import pandas as pd
import numpy as np
import piheaan as heaan
from piheaan.math import approx
import joblib

app = FastAPI()

# 정적 파일 경로 설정
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def read_root():
    return {"message": "Welcome to the HEAAN API"}

# Favicon 경로 설정
@app.get("/favicon.ico")
def favicon():
    return "Favicon not found", 404

# 동형암호화 관련 설정
params = heaan.ParameterPreset.FGb
context = heaan.make_context(params)
heaan.make_bootstrappable(context)

# Load pre-existing key
key_file_path = "./keys"

sk = heaan.SecretKey(context, key_file_path + "/secretkey.bin")  # load sk
pk = heaan.KeyPack(context, key_file_path + "/")  # load pk
pk.load_enc_key()
pk.load_mult_key()

eval = heaan.HomEvaluator(context, pk)
dec = heaan.Decryptor(context)
enc = heaan.Encryptor(context)

log_slots = 15
num_slots = 2**log_slots

# Normalize function (same as in your model code)
def normalize_data(arr):
    S = 0
    for i in range(len(arr)):
        S += arr[i]
    return [arr[i] / S for i in range(len(arr))]

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
def predict(data: Data):
    try:
        
        # 데이터를 DataFrame으로 변환
        input_data = pd.DataFrame([data.dict().values()], columns=data.dict().keys())
        
        # 정규화할 열 목록
        columns_to_normalize = [
            'bmi', 'forehead_circumference', 'neck_circumference', 'armpit_circumference',
            'bust', 'rib_cage', 'waist_circumference', 'iliac_circumference', 'femur_circumference',
            'urinenighttime_urination'
        ]

        # 정규화
        for col in columns_to_normalize:
            input_data[col] = normalize_data(input_data[col].values)

        # 데이터 암호화
        msg_X = heaan.Message(log_slots)
        ctxt_X = heaan.Ciphertext(context)
        for i, col in enumerate(input_data.columns):
            for j in range(len(input_data)):
                msg_X[len(input_data) * i + j] = input_data[col].iloc[j]
        enc.encrypt(msg_X, pk, ctxt_X)

        # 초기 beta 값 설정 (여기서는 임의로 설정, 실제 모델의 beta 값 사용 필요)
        beta = 2 * np.random.rand(32) - 1
        msg_beta = heaan.Message(log_slots)
        ctxt_beta = heaan.Ciphertext(context)
        for i in range(31):
            for j in range(len(input_data)):
                msg_beta[len(input_data) * i + j] = beta[i + 1]
        for j in range(len(input_data)):
            msg_beta[31 * len(input_data) + j] = beta[0]
        enc.encrypt(msg_beta, pk, ctxt_beta)

        # 로지스틱 회귀 예측 함수 호출 (동형암호화된 모델 사용)
        ctxt_result = compute_sigmoid(ctxt_X, ctxt_beta, len(input_data), log_slots, eval, context, num_slots)

        # 결과 복호화
        res = heaan.Message(log_slots)
        dec.decrypt(ctxt_result, sk, res)
        cnt = 0
        for i in range(len(input_data)):
            if res[i].real >= 0.6:
                prediction = 1
            else:
                prediction = 0
        
        return {
            "prediction": prediction,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Compute sigmoid function (as in your model code)
def compute_sigmoid(ctxt_X, ctxt_beta, n, log_slots, eval, context, num_slots):
    ctxt_rot = heaan.Ciphertext(context)
    ctxt_tmp = heaan.Ciphertext(context)
    
    # beta0
    ctxt_beta0 = heaan.Ciphertext(context)
    eval.left_rotate(ctxt_beta, 8 * n, ctxt_beta0)
    
    # compute x * beta + beta0
    ctxt_tmp = heaan.Ciphertext(context)
    eval.mult(ctxt_beta, ctxt_X, ctxt_tmp)
    
    for i in range(3):
        eval.left_rotate(ctxt_tmp, n * 2**(2 - i), ctxt_rot)
        eval.add(ctxt_tmp, ctxt_rot, ctxt_tmp)
    eval.add(ctxt_tmp, ctxt_beta0, ctxt_tmp)
    
    msg_mask = heaan.Message(log_slots)
    for i in range(n):
        msg_mask[i] = 1
    eval.mult(ctxt_tmp, msg_mask, ctxt_tmp)
    
    # compute sigmoid
    approx.sigmoid(eval, ctxt_tmp, ctxt_tmp, 8.0)
    eval.bootstrap(ctxt_tmp, ctxt_tmp)
    msg_mask = heaan.Message(log_slots)
    for i in range(n, num_slots):
        msg_mask[i] = 0.5
    eval.sub(ctxt_tmp, msg_mask, ctxt_tmp)
    
    return ctxt_tmp

# Run server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
