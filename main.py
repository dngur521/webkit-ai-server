# main.py
import re
import os
import asyncio
import shutil
import uuid
from pathlib import Path
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from typing import List, Optional

# --- 가비지 컬렉터 및 PyTorch 임포트 ---
import gc
import torch
# ---

import aiofiles # 비동기 파일 처리
from dotenv import load_dotenv # .env 파일 로드
from fastapi import FastAPI, Request, Form, File, UploadFile, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware # @CrossOrigin 대체
from pydantic import BaseModel # Java의 Record/DTO 대체

# --- 라이브러리 임포트 ---
from faster_whisper import WhisperModel
from llama_cpp import Llama
# ---

# --- 2. 설정 및 앱 초기화 ---

load_dotenv()  # .env 파일 로드
app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# .env 파일에서 Llama 모델 경로 읽기
LLAMA_MODEL_PATH = os.getenv("LLAMA_MODEL_PATH")
if not LLAMA_MODEL_PATH:
    print("치명적 오류: .env 파일에 LLAMA_MODEL_PATH가 설정되지 않았습니다.")
    exit(1)

# 임시 폴더 설정
TEMP_DIR = Path(os.getenv("java.io.tmpdir", os.getcwd())) / "fastapi-temp"
PARTIAL_SUMMARY_DIR = TEMP_DIR / "partial-summaries"

# --- 3. [핵심] 모델을 전역 변수로 선언하고 startup 시 로드 ---

whisper_model: Optional[WhisperModel] = None
llama_model: Optional[Llama] = None

@app.on_event("startup")
async def on_startup():
    global whisper_model, llama_model
    
    # 임시 폴더 생성
    os.makedirs(TEMP_DIR, exist_ok=True)
    os.makedirs(PARTIAL_SUMMARY_DIR, exist_ok=True)
    print(f"임시 파일 폴더: {TEMP_DIR.resolve()}")
    print(f"중간 요약 저장 폴더: {PARTIAL_SUMMARY_DIR.resolve()}")

    # 1. Whisper 모델 로드 (서버 시작 시 1회)
    try:
        print("[Whisper] 모델 로드 시작 (medium, cuda)...")
        # "medium" 모델을 HuggingFace에서 자동 다운로드 및 로드
        whisper_model = WhisperModel("medium", device="cuda", compute_type="float16")
        print("[Whisper] 모델 로드 완료.")
    except Exception as e:
        print(f"치명적 오류: Whisper 모델 로드 실패: {e}")
        # (모델 다운로드 실패 또는 VRAM 부족 시 여기서 서버가 중지될 수 있음)

    # 2. Llama 모델 로드 (서버 시작 시 1회)
    try:
        print(f"[Llama] 모델 로드 시작: {LLAMA_MODEL_PATH}")
        llama_model = Llama(
            model_path=LLAMA_MODEL_PATH,
            n_gpu_layers=30,  # -1 = 가능한 만큼 GPU에 올림
            n_ctx=4096,
            n_threads=8,
            n_batch=512,
            verbose=True # 시작 시 로그 확인
        )
        print("[Llama] 모델 로드 완료.")
    except Exception as e:
        print(f"치명적 오류: Llama 모델 로드 실패: {e}")

# VRAM 해제 로직 (서버 종료 시)
@app.on_event("shutdown")
async def on_shutdown():
    global whisper_model, llama_model
    try:
        print("서버 종료... AI 모델 언로드 중...")
        if whisper_model: del whisper_model
        if llama_model: del llama_model
        gc.collect()
        torch.cuda.empty_cache()
        print("AI 모델 VRAM 언로드 완료.")
    except Exception as e:
        print(f"모델 언로드 중 오류: {e}")

# --- 4. DTO 정의 ---
class SttResponse(BaseModel):
    text: Optional[str] = None
    error: Optional[str] = None
    transcriptId: Optional[str] = None

class RetryRequest(BaseModel):
    startTime: str
    transcriptId: str

class SimpleSummaryResponse(BaseModel):
    text: Optional[str] = None
    error: Optional[str] = None

# --- 5. STT 및 Llama 실행 헬퍼 함수 ---
# (모델을 인자로 받지 않고, 전역 변수 whisper_model, llama_model을 사용)

async def run_stt_on_file(audio_file: UploadFile) -> Optional[str]:
    """
    업로드된 오디오 파일 1개를 STT 처리 (전역 모델 사용)
    """
    global whisper_model
    if not whisper_model:
        raise HTTPException(status_code=503, detail="Whisper 모델이 준비되지 않았습니다.")
    
    temp_file_path = None
    try:
        temp_file_path = TEMP_DIR / f"stt-in-{uuid.uuid4()}.webm"
        async with aiofiles.open(temp_file_path, 'wb') as f:
            content = await audio_file.read()
            await f.write(content)

        def transcribe_sync():
            segments, info = whisper_model.transcribe(str(temp_file_path), language="ko", beam_size=5)
            return " ".join([segment.text.strip() for segment in segments if segment.text.strip()])

        print(f"Whisper STT 시작: {audio_file.filename}")
        transcript_result = await asyncio.to_thread(transcribe_sync)
        print(f"Whisper STT 완료: {audio_file.filename}")
        
        if not transcript_result.strip():
            print(f"Whisper 결과 없음 ({audio_file.filename})")
            return None

        filename_no_ext = (audio_file.filename or "").rsplit('.', 1)[0]
        return f"{filename_no_ext}: {transcript_result}"
    except Exception as e:
        print(f"오류: {audio_file.filename} STT 처리 중 예외: {e}")
        return None
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try: os.remove(temp_file_path)
            except Exception as e: print(f"오류: 임시 파일 삭제 실패: {e}")

async def run_stt(audio_files: List[UploadFile]) -> str:
    """
    오디오 파일 목록을 병렬로 STT 처리
    """
    global whisper_model
    if not whisper_model:
        raise HTTPException(status_code=503, detail="Whisper 모델이 준비되지 않았습니다.")
    tasks = [run_stt_on_file(file) for file in audio_files] # whisper_model 인자 제거
    results = await asyncio.gather(*tasks)
    return "\n".join(r for r in results if r)

async def get_summary_from_llama(transcript: str, meeting_start_time_str: str, chunk_start_time_str: Optional[str] = None) -> str:
    """
    Llama 모델을 실행하여 '중간 요약' (테이블) 생성 (전역 모델 사용)
    """
    global llama_model
    if not llama_model:
        raise HTTPException(status_code=503, detail="Llama 모델이 준비되지 않았습니다.")

    # 1. 회의 시작 시간 (KST)
    try:
        utc_meeting_start = datetime.fromisoformat(meeting_start_time_str)
        korean_meeting_start = utc_meeting_start.astimezone(ZoneInfo("Asia/Seoul"))
        formatted_meeting_start_time = korean_meeting_start.strftime("%H:%M:%S")
    except Exception: 
        formatted_meeting_start_time = "(시간 정보 오류)"
        korean_meeting_start = datetime.now(ZoneInfo("Asia/Seoul")) # Fallback

    # 2. [추가] 청크 시작 시간 (KST) 및 오프셋 계산
    if not chunk_start_time_str:
        chunk_start_time_str = meeting_start_time_str # Fallback (첫 청크)
            
    try:
        utc_chunk_start = datetime.fromisoformat(chunk_start_time_str)
        korean_chunk_start = utc_chunk_start.astimezone(ZoneInfo("Asia/Seoul"))
        # [추가] 회의 시작으로부터 이 청크까지 몇 초가 지났는지 계산
        time_offset_seconds = (korean_chunk_start - korean_meeting_start).total_seconds()
        if time_offset_seconds < 0:
            time_offset_seconds = 0
    except Exception:
        time_offset_seconds = 0.0

    system_prompt = (
        f"당신은 회의록을 분석하여 마크다운 테이블 형식으로 요약하는 전문 비서입니다.\n"
        f"이 회의는 {formatted_meeting_start_time}에 시작되었습니다.\n"
        f"현재 제공되는 텍스트(회의록 청크)는 회의 시작 후 약 {int(time_offset_seconds)}초가 지난 시점부터의 내용입니다.\n"
        "아래 텍스트의 타임스탬프는 이 청크 시작(00:00) 기준입니다. 이를 실제 시간으로 계산해야 합니다.\n\n"
        "### 주요 규칙:\n"
        "1. **시간 계산:** '회의 시작 시간'({formatted_meeting_start_time}) + '청크 오프셋'({int(time_offset_seconds)}초) + '청크 내 타임스탬프'를 더하여 실제 시간을 계산하세요.\n"
        "2. **발언 병합:** 같은 발언자가 연속해서 말하는 경우, 내용을 요약하여 **하나의 행으로 합쳐야 합니다.** 시간 구간은 합쳐진 발언의 시작 시간과 끝 시간으로 표시합니다.\n"
        "3. **정확한 형식:** 반드시 아래 마크다운 테이블 형식에 맞춰 응답하고, 다른 설명은 절대 추가하지 마세요. 그리고 시간 구간에 소숫점은 절대 추가하지 마세요.\n\n"
        "### 예시:\n"
        "| 시간 구간           | 발언자 | 핵심 내용                  |\n"
        "|---------------------|--------|----------------------------|\n"
        "| 14:30:16-14:30:45 | 이영희 | 다음 주까지 기획서 마무리를 제안하고, UI 디자인 파트를 담당하겠다고 말함. |\n\n"
    )
    user_prompt = f"아래 회의록을 분석하여 요약 테이블을 생성해 주세요:\n\n---\n{transcript}"
    
    final_prompt = (
        "/no_think <|im_start|>system\n" +
        system_prompt +
        "<|im_end|>\n<|im_start|>user\n" +
        user_prompt +
        "<|im_end|>\n<|im_start|>assistant\n"
    )
    print(f"Llama 모델로 '중간 요약' 생성 시작 (오프셋: {int(time_offset_seconds)}초)")

    def create_completion_sync():
        return llama_model.create_completion(
            prompt=final_prompt, temperature=0.5, max_tokens=2048, stream=False
        )
    output = await asyncio.to_thread(create_completion_sync)
    summary = output['choices'][0]['text']
    summary = re.sub(r"<think>.*?</think>", "", summary, flags=re.DOTALL).strip()
    print("Llama '중간 요약' 생성 완료.")
    return summary or ""

async def run_simple_summary(text_to_summarize: str) -> str:
    """
    Llama 모델을 실행하여 '단순 요약' 생성 (전역 모델 사용)
    """
    global llama_model
    if not llama_model:
        raise HTTPException(status_code=503, detail="Llama 모델이 준비되지 않았습니다.")

    # 단순 요약을 위한 프롬프트
    system_prompt = (
        "당신은 유용한 AI 어시스턴트입니다.\n"
        "사용자가 제공한 텍스트를 핵심 내용만 뽑아서 **마크다운 형식**, **Notion 스타일**로 주제별로 문단을 나누고 중요 내용은 글머리 기호(-)와 💡, 📅, 👤 같은 이모지를 사용하여 깔끔하게 정리하세요.\n"
        "**중요:** 응답을 마크다운 코드 블록(```)으로 절대 감싸지 마세요. 다른 설명 없이 요약된 마크다운 내용 원본으로 바로 시작해야 합니다.\n"
    )
    user_prompt = f"아래 텍스트를 요약해 주세요:\n\n---\n{text_to_summarize}"
    
    final_prompt = (
        "/no_think <|im_start|>system\n" +
        system_prompt +
        "<|im_end|>\n<|im_start|>user\n" +
        user_prompt +
        "<|im_end|>\n<|im_start|>assistant\n"
    )
    print("Llama 모델로 '단순 요약' 생성 시작...")

    def create_completion_sync():
        return llama_model.create_completion(
            prompt=final_prompt, 
            temperature=0.5,
            max_tokens=1024, # 요약에 필요한 토큰
            stream=False
        )
    output = await asyncio.to_thread(create_completion_sync)
    summary = output['choices'][0]['text']
    summary = re.sub(r"<think>.*?</think>", "", summary, flags=re.DOTALL).strip()
    print("Llama '단순 요약' 생성 완료.")
    return summary or "요약 내용을 생성하지 못했습니다."

async def get_final_report_from_llama(all_partial_summaries: str, start_time_str: str, end_time_str: Optional[str] = None) -> str:
    """
    Llama 모델을 실행하여 '최종 보고서' (Notion 스타일) 생성 (전역 모델 사용)
    [수정] end_time_str 인자 추가
    """
    global llama_model
    if not llama_model:
        raise HTTPException(status_code=503, detail="Llama 모델이 준비되지 않았습니다.")

    # 1. 회의 시작 시간
    try:
        utc_start_time = datetime.fromisoformat(start_time_str)
        korean_start_time = utc_start_time.astimezone(ZoneInfo("Asia/Seoul"))
        formatted_start_time = korean_start_time.strftime("%Y-%m-%d %H:%M:%S")
        formatted_start_hhmmss = korean_start_time.strftime("%H:%M:%S")
    except Exception: 
        formatted_start_time = "(시작 시간 오류)"
        formatted_start_hhmmss = "HH:mm:ss"

    # 2. [추가] 회의 종료 시간
    if not end_time_str:
        end_time_str = datetime.now().isoformat() # fallback
            
    try:
        utc_end_time = datetime.fromisoformat(end_time_str)
        korean_end_time = utc_end_time.astimezone(ZoneInfo("Asia/Seoul"))
        formatted_end_hhmmss = korean_end_time.strftime("%H:%M:%S")
    except Exception:
        formatted_end_hhmmss = "HH:mm:ss"


    system_prompt = (
        f"당신은 회의의 중간 요약본들을 취합하여 하나의 최종 보고서를 작성하는 전문 비서입니다.\n"
        f"이 회의는 {formatted_start_time} (KST)에 시작되었습니다.\n\n"
        "### 지시사항:\n"
        "1.  **보고서 재구성:** 제공된 모든 중간 요약(마크다운 테이블 형식) 내용을 바탕으로, 하나의 일관된 최종 보고서를 **서술형**으로 작성하세요.\n"
        "2.  **시간 정보 통합:** 중간 요약에 있는 시간 정보를 활용하여 회의 내용을 시간 순서대로 자연스럽게 기술하세요. 단, 최종 보고서 본문에는 **개별 발언의 타임스탬프를 절대 포함하지 마세요.**\n"
        "3.  **Notion 스타일:** 주제별로 문단을 나누고 중요 내용은 글머리 기호(-)를 사용하여 깔끔하게 정리하세요.\n"
        "4.  **필수 섹션:** '주요 결정 사항'과 '실행 항목(Action Items)' 섹션을 반드시 포함하고, 관련 내용을 각 섹션 아래에 명확히 요약하세요.\n"
        "5.  **시간 범위 명시 (가장 중요):**\n"
        f"    * 보고서의 가장 첫 줄은 반드시 `## 회의 주요 내용 ({formatted_start_hhmmss} ~ {formatted_end_hhmmss})` 형식이어야 합니다.\n"
        f"    * 괄호 안의 시작 시간({formatted_start_hhmmss})과 종료 시간({formatted_end_hhmmss})은 **정확히** 제공된 값을 사용해야 합니다.\n"
        "    * 시간은 **반드시 'HH:mm:ss' 형식**을 따라야 합니다. (예: 15:30:05)\n"
        "    * **시간(HH)은 00~23 사이, 분(mm)과 초(ss)는 00~59 사이의 값이어야 합니다.** 절대 다른 형식이나 범위를 벗어난 값을 사용하지 마세요.\n"
        "6.  **마크다운 테이블 금지:** 최종 보고서에는 마크다운 테이블 형식을 절대 사용하지 마세요.\n\n"

        "### 최종 보고서 출력 형식 예시:\n"
        f"## 회의 주요 내용 ({formatted_start_hhmmss} ~ {formatted_end_hhmmss})\n"
        "- (주제 1에 대한 논의 내용을 서술형으로 작성...)\n"
        "- (주제 2에 대한 논의 내용을 글머리 기호로 요약...)\n\n"
        "## 주요 결정 사항\n"
        "- (결정된 사항 1...)\n"
        "- (결정된 사항 2...)\n\n"
        "## 실행 항목 (Action Items)\n"
        "- (담당자: 마감일 - 실행할 내용...)\n"
    )
    user_prompt = f"아래는 회의의 중간 요약본들입니다. 이 내용을 바탕으로 '최종 보고서'를 작성해주세요:\n\n---\n{all_partial_summaries}"

    final_prompt = (
        "/no_think <|im_start|>system\n" +
        system_prompt +
        "<|im_end|>\n<|im_start|>user\n" +
        user_prompt +
        "<|im_end|>\n<|im_start|>assistant\n"
    )
    print("Llama 모델로 '최종 요약' 생성 시작...")

    def create_completion_sync():
        return llama_model.create_completion(
            prompt=final_prompt, temperature=0.5, max_tokens=2048
        )
    output = await asyncio.to_thread(create_completion_sync)
    summary = output['choices'][0]['text']
    summary = re.sub(r"<think>.*?</think>", "", summary, flags=re.DOTALL).strip()
    print("Llama '최종 요약' 생성 완료.")
    return summary or ""

async def generate_final_summary(meeting_id: str, start_time: str, end_time: Optional[str] = None) -> str:
    """
    저장된 모든 중간 요약 파일을 읽어 '최종 요약' 생성 (전역 Llama 모델 사용)
    [수정] end_time 인자 추가
    """
    global llama_model
    if not llama_model:
        raise HTTPException(status_code=503, detail="Llama 모델이 준비되지 않았습니다.")
        
    meeting_dir = PARTIAL_SUMMARY_DIR / meeting_id
    if not os.path.exists(meeting_dir):
        raise HTTPException(status_code=404, detail=f"요약 폴더를 찾을 수 없습니다: {meeting_id}")

    summary_files = sorted(meeting_dir.glob("*.txt"))
    all_summaries = []
    for file_path in summary_files:
        try:
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                all_summaries.append(await f.read())
        except Exception as e:
            all_summaries.append("") # [수정] 오류 시 빈 문자열 추가

    all_summaries_text = "\n\n---\n\n".join(all_summaries)
    if not all_summaries_text.strip():
        return "요약할 내용이 없습니다."

    # [수정] end_time을 get_final_report_from_llama로 전달
    return await get_final_report_from_llama(all_summaries_text, start_time, end_time)

# --- 6. API 엔드포인트 ---

@app.post("/summary", response_model=SimpleSummaryResponse)
async def handle_simple_summary(text: str = Form(...)):
    """
    제공된 텍스트를 받아 단순 요약을 반환합니다.
    """
    
    # 전역 모델이 로드되었는지 확인
    if not llama_model:
        return SimpleSummaryResponse(error="AI 모델이 아직 준비되지 않았습니다.")
    
    if not text or not text.strip():
        return SimpleSummaryResponse(error="요약할 'text' 내용이 없습니다.")
        
    try:
        # 1. 단순 요약 헬퍼 함수 호출
        summary_result = await run_simple_summary(text)
        
        # 2. 요약 결과 반환
        return SimpleSummaryResponse(text=summary_result)

    except Exception as e:
        print(f"오류: /summary 엔드포인트 처리 중 예외 발생: {e}")
        import traceback
        traceback.print_exc()
        return SimpleSummaryResponse(error=f"서버 내부 오류: {e}")

@app.post("/process-audio-chunk", response_model=SttResponse)
async def handle_audio_chunk(
    meetingId: str = Form(...),
    startTime: str = Form(...),
    isFinal: bool = Form(...),
    audio_files: List[UploadFile] = File(...),
    chunkStartTime: Optional[str] = Form(None), 
    endTime: Optional[str] = Form(None)
):
    # 전역 모델이 로드되었는지 확인
    if not whisper_model or not llama_model:
        raise HTTPException(status_code=503, detail="AI 모델이 아직 준비되지 않았습니다. 서버 시작 로그를 확인하세요.")
        
    try:
        # 1. STT 실행 (전역 모델 사용)
        full_transcript = await run_stt(audio_files) # whisper_model 인자 제거

        # STT 결과 없음 처리
        if not full_transcript.strip():
            print(f"STT 결과가 비어있음 (MeetingID: {meetingId}, isFinal: {isFinal})")
            if isFinal:
                pass 
            else:
                return SttResponse()

        partial_summary = ""
        if full_transcript.strip():
            # 2. 중간 요약 생성 (전역 모델 사용)
            try:
                partial_summary = await get_summary_from_llama(full_transcript, startTime, chunkStartTime)
            except Exception as e:
                print(f"오류: 중간 요약 생성 실패 (MeetingID: {meetingId}, isFinal: {isFinal}): {e}")
                if not isFinal:
                    return SttResponse()
        
        # 3. 중간 요약 파일 저장 (동일)
        meeting_dir = PARTIAL_SUMMARY_DIR / meetingId
        if partial_summary and partial_summary.strip():
            try:
                os.makedirs(meeting_dir, exist_ok=True)
                part_file_name = f"{int(datetime.now().timestamp())}_summary.txt"
                summary_file_path = meeting_dir / part_file_name
                async with aiofiles.open(summary_file_path, 'w', encoding='utf-8') as f:
                    await f.write(partial_summary)
                print(f"중간 요약 저장: {summary_file_path}")
            except Exception as e:
                print(f"오류: 중간 요약 저장 실패 (MeetingID: {meetingId}): {e}")
                if not isFinal:
                    return SttResponse()
                print("최종 요약 처리 중 중간 요약 저장 실패 발생.")
        else:
            print(f"생성된 중간 요약 내용이 없어 파일을 저장하지 않습니다. (MeetingID: {meetingId})")

        # 4. isFinal 플래그에 따라 분기
        if isFinal:
            print(f"최종 요약 생성을 시작합니다 (MeetingID: {meetingId})")
            try:
                # 5. 최종 요약 생성 (전역 모델 사용)
                final_summary = await generate_final_summary(meetingId, startTime, endTime)

                # 6. 중간 요약 파일들 삭제 (동일)
                try:
                    if os.path.exists(meeting_dir):
                        shutil.rmtree(meeting_dir)
                        print(f"중간 요약 파일 삭제 완료: {meeting_dir}")
                except Exception as e:
                    print(f"오류: 중간 요약 파일 삭제 중 오류 발생 (MeetingID: {meetingId}): {e}")

                # 7. '최종 요약' 반환
                return SttResponse(text=final_summary)

            except Exception as e:
                print(f"오류: 최종 요약 생성 중 심각한 오류 발생 (MeetingID: {meetingId}): {e}")
                import traceback
                traceback.print_exc() # 스택 트레이스 출력
                return SttResponse(error=f"최종 요약 생성 실패: {e}", transcriptId=meetingId)
        else:
            # [중간 요약] 요청인 경우, 빈 성공 응답 반환
            return SttResponse()

    except Exception as e:
        # 예상치 못한 최상위 예외 처리
        print(f"오류: handle_audio_chunk 처리 중 알 수 없는 오류 (MeetingID: {meetingId}): {e}")
        import traceback
        traceback.print_exc()
        return SttResponse(error=f"서버 내부 오류: {e}", transcriptId=meetingId if isFinal else None)


@app.post("/retry-final-summary", response_model=SttResponse)
async def handle_retry(retry_request: RetryRequest):
    """
    저장된 중간 요약 파일들로 '최종 요약' 생성을 재시도
    """
    global llama_model # 전역 모델 사용
    meeting_id = retry_request.transcriptId
    start_time = retry_request.startTime
    meeting_dir = PARTIAL_SUMMARY_DIR / meeting_id

    if not os.path.exists(meeting_dir):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail="저장된 중간 요약 폴더를 찾을 수 없습니다."
        )
    
    if not llama_model:
        raise HTTPException(status_code=503, detail="Llama 모델이 준비되지 않았습니다.")

    try:
        # 1. 최종 요약 생성 (전역 모델 사용)
        # 재시도 시점의 시간을 endTime으로 전달
        end_time = datetime.now().isoformat()
        final_summary = await generate_final_summary(meeting_id, start_time, end_time)

        # 2. 중간 요약 파일 삭제
        try:
            shutil.rmtree(meeting_dir)
            print(f"중간 요약 파일 삭제 완료 (재시도): {meeting_dir}")
        except Exception as e:
            print(f"오류: 중간 요약 파일 삭제 중 오류 발생 (재시도) (MeetingID: {meeting_id}): {e}")
        
        # 3. 성공 응답
        return SttResponse(text=final_summary)

    except Exception as e:
        print(f"오류: 최종 요약 재시도 중 심각한 오류 발생 (MeetingID: {meeting_id}): {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"최종 요약 재시도 실패: {e}"
        )
    

# --- 7. (선택) uvicorn으로 바로 실행 ---
if __name__ == "__main__":
    import uvicorn
    # Python main.py를 직접 실행할 경우 (개발용)
    print(f"Llama 모델 경로 확인: {LLAMA_MODEL_PATH}")
    uvicorn.run(app, host="0.0.0.0", port=8081, log_level="info")