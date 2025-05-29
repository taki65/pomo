# pip install torch==2.2.2+cpu torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
# pip install transformers==4.33.2
from transformers import pipeline
import random

def generate_cheer_message():
    generator = pipeline(
        'text-generation',
        model='skt/kogpt2-base-v2',
        tokenizer='skt/kogpt2-base-v2',
        framework='pt',  # PyTorch 강제 지정
        device=-1
    )

    prompts = [
    "오늘 하루 목표를 달성하신 것을 축하드립니다!",
    "훌륭한 집중력 입니다, 조금만 더 해볼까요?",
    "오늘도 멋진 하루 보내세요!",
    "포기하지 말고 계속 나아가요!",
    "작은 성취가 큰 변화를 만듭니다!",
    "정말 잘했어요! 지금 이 순간이 성장의 시작입니다. 조금만 더 힘내봐요!",
    "오늘의 노력은 내일의 성공을 만듭니다. 계속해서 나아가요!",
    "당신의 꾸준함이 빛나고 있어요. 이 페이스를 유지해봐요!",
    "한 걸음 한 걸음이 소중합니다. 포기하지 말고 함께 달려가요!",
    "이미 멋진 성과를 이루었어요! 이제 다음 목표를 향해 가볼까요?",
    "자신을 믿고 끝까지 도전하세요! 더 큰 성취가 기다리고 있습니다.",
    "지금 이 순간에도 성장하고 있어요. 계속 전진합시다!",
    "성공은 꾸준함의 결과입니다. 조금 더 힘내면 더 큰 기쁨이 올 거예요!",
    "매일의 작은 승리가 모여 큰 기적을 만듭니다. 함께 해요!",
    "당신의 열정이 정말 대단해요. 오늘도 파이팅입니다!"
]

    prompt = random.choice(prompts)
    result = generator(prompt, max_length=40, num_return_sequences=1)
    message = result[0]['generated_text']

    return message

if __name__ == "__main__":
    msg = generate_cheer_message()
    print(msg)