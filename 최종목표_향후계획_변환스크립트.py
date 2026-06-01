"""
최종목표_향후계획.md → PowerPoint 자동 변환 스크립트
프로젝트: 시니어 온디바이스 헬스케어 지식증류 LLM
"""

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.dml.color import RGBColor
import re

# 색상 정의
COLOR_PRIMARY = RGBColor(0, 102, 204)      # 진 파란색
COLOR_ACCENT = RGBColor(255, 149, 0)      # 주황색
COLOR_SUCCESS = RGBColor(0, 176, 80)      # 초록색
COLOR_WARNING = RGBColor(255, 0, 0)       # 빨간색
COLOR_NEUTRAL = RGBColor(153, 153, 153)   # 회색
COLOR_WHITE = RGBColor(255, 255, 255)     # 흰색
COLOR_LIGHT_GRAY = RGBColor(242, 242, 242) # 연한 회색

def create_presentation():
    """PPT 프레젠테이션 생성 및 기본 설정"""
    prs = Presentation()
    prs.slide_width = Inches(10)
    prs.slide_height = Inches(7.5)
    return prs

def add_title_slide(prs):
    """Slide 1: Cover (표지)"""
    slide = prs.slides.add_slide(prs.slide_layouts[6])  # 빈 슬라이드
    background = slide.background
    fill = background.fill
    fill.solid()
    fill.fore_color.rgb = COLOR_PRIMARY
    
    # 제목
    title_box = slide.shapes.add_textbox(Inches(0.5), Inches(1.5), Inches(9), Inches(1.5))
    title_frame = title_box.text_frame
    title_frame.word_wrap = True
    title_p = title_frame.paragraphs[0]
    title_p.text = "13B 경량 모델 + 공공 의료 데이터로\n시니어 온디바이스 헬스케어 서비스 실현"
    title_p.font.size = Pt(40)
    title_p.font.bold = True
    title_p.font.color.rgb = COLOR_WHITE
    title_p.alignment = PP_ALIGN.CENTER
    
    # 목표 박스들
    goals = [
        ("모델", "13B 지식증류 모델\n(온디바이스 실행)", COLOR_PRIMARY),
        ("데이터", "공공의료 데이터\n기반 의료 Q&A", COLOR_SUCCESS),
        ("정확도", "90% 이상\n(모든 의료 상황 커버)", COLOR_ACCENT),
        ("배포", "모바일 앱 + 웹\n(완전 오프라인 지원)", RGBColor(128, 0, 255))
    ]
    
    box_width = 2
    start_x = 0.5
    start_y = 3.5
    
    for idx, (label, content, color) in enumerate(goals):
        x = start_x + (idx * (box_width + 0.2))
        shape = slide.shapes.add_shape(1, Inches(x), Inches(start_y), Inches(box_width), Inches(2.5))
        shape.fill.solid()
        shape.fill.fore_color.rgb = color
        shape.line.color.rgb = COLOR_WHITE
        shape.line.width = Pt(2)
        
        text_frame = shape.text_frame
        text_frame.word_wrap = True
        text_frame.margin_bottom = Inches(0.1)
        text_frame.margin_top = Inches(0.1)
        
        p = text_frame.paragraphs[0]
        p.text = label
        p.font.size = Pt(14)
        p.font.bold = True
        p.font.color.rgb = COLOR_WHITE
        p.alignment = PP_ALIGN.CENTER
        
        p2 = text_frame.add_paragraph()
        p2.text = content
        p2.font.size = Pt(11)
        p2.font.color.rgb = COLOR_WHITE
        p2.alignment = PP_ALIGN.CENTER
        p2.space_before = Pt(6)
    
    # 하단 메시지
    footer_box = slide.shapes.add_textbox(Inches(0.5), Inches(6.5), Inches(9), Inches(0.8))
    footer_frame = footer_box.text_frame
    footer_frame.word_wrap = True
    fp = footer_frame.paragraphs[0]
    fp.text = "→ 통신 없이도 시니어가 안전하게 의료 정보에 접근하는 서비스 실현"
    fp.font.size = Pt(16)
    fp.font.bold = True
    fp.font.color.rgb = COLOR_ACCENT
    fp.alignment = PP_ALIGN.CENTER

def add_timeline_slide(prs):
    """Slide 2: 5-Step 타임라인 (Pilot → Beta → Full)"""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    background = slide.background
    fill = background.fill
    fill.solid()
    fill.fore_color.rgb = COLOR_WHITE
    
    # 제목
    title_box = slide.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(9), Inches(0.8))
    title_frame = title_box.text_frame
    tp = title_frame.paragraphs[0]
    tp.text = "[방법론] 데이터 수집 → 모델 학습 → 안전 검증 → 배포의 4단계로 단계별 목표 달성"
    tp.font.size = Pt(18)
    tp.font.bold = True
    tp.font.color.rgb = COLOR_PRIMARY
    title_frame.word_wrap = True
    
    # 3단계 타임라인
    stages = [
        {
            "name": "Pilot",
            "period": "1~2개월",
            "data": "500개 Q&A 쌍",
            "query": "50개 대표 쿼리",
            "goal": "기술 가능성 검증",
            "color": RGBColor(200, 220, 255)
        },
        {
            "name": "Beta",
            "period": "3~4개월",
            "data": "3,000개 Q&A 쌍",
            "query": "500개 쿼리",
            "goal": "정확도 85% 이상 달성",
            "color": RGBColor(100, 180, 255)
        },
        {
            "name": "Full",
            "period": "5~6개월",
            "data": "10,000개 Q&A 쌍",
            "query": "2,000개 쿼리",
            "goal": "정확도 90% 이상 & Gate 통과",
            "color": COLOR_PRIMARY
        }
    ]
    
    box_width = 2.8
    start_x = 0.5
    start_y = 1.3
    
    for idx, stage in enumerate(stages):
        x = start_x + (idx * (box_width + 0.2))
        
        # 박스
        shape = slide.shapes.add_shape(1, Inches(x), Inches(start_y), Inches(box_width), Inches(5))
        shape.fill.solid()
        shape.fill.fore_color.rgb = stage["color"]
        shape.line.color.rgb = COLOR_PRIMARY
        shape.line.width = Pt(2)
        
        text_frame = shape.text_frame
        text_frame.word_wrap = True
        text_frame.vertical_anchor = MSO_ANCHOR.TOP
        text_frame.margin_left = Inches(0.15)
        text_frame.margin_top = Inches(0.15)
        text_frame.margin_right = Inches(0.15)
        
        # 단계명
        p = text_frame.paragraphs[0]
        p.text = stage["name"]
        p.font.size = Pt(20)
        p.font.bold = True
        p.font.color.rgb = COLOR_PRIMARY
        p.alignment = PP_ALIGN.CENTER
        
        # 기간
        p = text_frame.add_paragraph()
        p.text = stage["period"]
        p.font.size = Pt(12)
        p.font.color.rgb = COLOR_NEUTRAL
        p.alignment = PP_ALIGN.CENTER
        p.space_before = Pt(3)
        
        # 구분선
        p = text_frame.add_paragraph()
        p.text = "━" * 12
        p.font.size = Pt(10)
        p.font.color.rgb = COLOR_NEUTRAL
        p.alignment = PP_ALIGN.CENTER
        p.space_before = Pt(6)
        
        # 데이터
        p = text_frame.add_paragraph()
        p.text = "데이터"
        p.font.size = Pt(11)
        p.font.bold = True
        p.font.color.rgb = COLOR_PRIMARY
        p.space_before = Pt(6)
        
        p = text_frame.add_paragraph()
        p.text = stage["data"]
        p.font.size = Pt(10)
        p.font.color.rgb = RGBColor(50, 50, 50)
        p.level = 1
        
        # 쿼리
        p = text_frame.add_paragraph()
        p.text = "질의"
        p.font.size = Pt(11)
        p.font.bold = True
        p.font.color.rgb = COLOR_PRIMARY
        p.space_before = Pt(3)
        
        p = text_frame.add_paragraph()
        p.text = stage["query"]
        p.font.size = Pt(10)
        p.font.color.rgb = RGBColor(50, 50, 50)
        p.level = 1
        
        # 목표
        p = text_frame.add_paragraph()
        p.text = "목표"
        p.font.size = Pt(11)
        p.font.bold = True
        p.font.color.rgb = COLOR_PRIMARY
        p.space_before = Pt(3)
        
        p = text_frame.add_paragraph()
        p.text = stage["goal"]
        p.font.size = Pt(10)
        p.font.color.rgb = RGBColor(50, 50, 50)
        p.level = 1
    
    # 하단 결론
    footer_box = slide.shapes.add_textbox(Inches(0.5), Inches(6.5), Inches(9), Inches(0.8))
    footer_frame = footer_box.text_frame
    footer_frame.word_wrap = True
    fp = footer_frame.paragraphs[0]
    fp.text = "→ 각 단계마다 정확도·응답 시간·위험 응답 여부를 측정해 다음 단계 진행 여부 결정"
    fp.font.size = Pt(12)
    fp.font.bold = True
    fp.font.color.rgb = COLOR_ACCENT
    fp.alignment = PP_ALIGN.CENTER

def add_table_slide(prs):
    """Slide 3: 데이터·모델·평가 목표 테이블"""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    background = slide.background
    fill = background.fill
    fill.solid()
    fill.fore_color.rgb = COLOR_WHITE
    
    # 제목
    title_box = slide.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(9), Inches(0.8))
    title_frame = title_box.text_frame
    tp = title_frame.paragraphs[0]
    tp.text = "[기술] Pilot·Beta·Full 각 단계별 데이터·모델·평가 목표 명확화"
    tp.font.size = Pt(16)
    tp.font.bold = True
    tp.font.color.rgb = COLOR_PRIMARY
    title_frame.word_wrap = True
    
    # 테이블 추가
    rows, cols = 7, 4
    left = Inches(0.5)
    top = Inches(1.3)
    width = Inches(9)
    height = Inches(4.5)
    
    table_shape = slide.shapes.add_table(rows, cols, left, top, width, height)
    table = table_shape.table
    
    # 헤더
    headers = ["항목", "Pilot", "Beta", "Full"]
    for col_idx, header in enumerate(headers):
        cell = table.cell(0, col_idx)
        cell.fill.solid()
        cell.fill.fore_color.rgb = COLOR_PRIMARY
        cell.text = header
        for paragraph in cell.text_frame.paragraphs:
            for run in paragraph.runs:
                run.font.size = Pt(14)
                run.font.bold = True
                run.font.color.rgb = COLOR_WHITE
            paragraph.alignment = PP_ALIGN.CENTER
    
    # 데이터
    data = [
        ["공공 의료 데이터", "500개 Q&A", "3,000개 Q&A", "10,000개 Q&A"],
        ["시니어 질문 패턴", "50개 쿼리", "500개 쿼리", "2,000개 쿼리"],
        ["의료 주제 범위", "주요 질환 5개", "주요 질환 20개", "모든 의료 상황"],
        ["정확도 목표", "70% 이상", "85% 이상", "90% 이상"],
        ["응답 시간", "<5초", "<3초", "<2초"],
        ["배포 환경", "테스트 환경", "베타 사용자(100명)", "모바일 앱 + 웹"]
    ]
    
    for row_idx, row_data in enumerate(data, start=1):
        for col_idx, cell_data in enumerate(row_data):
            cell = table.cell(row_idx, col_idx)
            
            # 배경색 (교차 색상)
            if row_idx % 2 == 0:
                cell.fill.solid()
                cell.fill.fore_color.rgb = COLOR_LIGHT_GRAY
            else:
                cell.fill.solid()
                cell.fill.fore_color.rgb = COLOR_WHITE
            
            # Full 열의 90% 셀 강조
            if col_idx == 3 and row_idx == 4:
                cell.fill.solid()
                cell.fill.fore_color.rgb = COLOR_ACCENT
            
            cell.text = cell_data
            for paragraph in cell.text_frame.paragraphs:
                for run in paragraph.runs:
                    run.font.size = Pt(11)
                    if col_idx == 3 and row_idx == 4:
                        run.font.bold = True
                        run.font.color.rgb = COLOR_WHITE
                paragraph.alignment = PP_ALIGN.CENTER
    
    # 하단 결론
    footer_box = slide.shapes.add_textbox(Inches(0.5), Inches(6.2), Inches(9), Inches(0.8))
    footer_frame = footer_box.text_frame
    footer_frame.word_wrap = True
    fp = footer_frame.paragraphs[0]
    fp.text = "→ 각 단계는 데이터 규모·정확도·응답 속도 목표가 명확하며, 미달 시 이전 단계로 재귀"
    fp.font.size = Pt(12)
    fp.font.bold = True
    fp.font.color.rgb = COLOR_ACCENT
    fp.alignment = PP_ALIGN.CENTER

def add_comparison_slide(prs):
    """Slide 4: 온디바이스 vs 클라우드 비교"""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    background = slide.background
    fill = background.fill
    fill.solid()
    fill.fore_color.rgb = COLOR_WHITE
    
    # 제목
    title_box = slide.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(9), Inches(0.8))
    title_frame = title_box.text_frame
    tp = title_frame.paragraphs[0]
    tp.text = "[안전] 온디바이스 오프라인 서비스 vs 클라우드 기반 서비스 비교"
    tp.font.size = Pt(16)
    tp.font.bold = True
    tp.font.color.rgb = COLOR_PRIMARY
    title_frame.word_wrap = True
    
    # 좌측: 온디바이스 (우리 방식)
    left_box = slide.shapes.add_shape(1, Inches(0.5), Inches(1.3), Inches(4.3), Inches(4.8))
    left_box.fill.solid()
    left_box.fill.fore_color.rgb = RGBColor(220, 250, 220)
    left_box.line.color.rgb = COLOR_SUCCESS
    left_box.line.width = Pt(3)
    
    left_frame = left_box.text_frame
    left_frame.word_wrap = True
    left_frame.margin_left = Inches(0.2)
    left_frame.margin_top = Inches(0.2)
    
    lp = left_frame.paragraphs[0]
    lp.text = "✓ 온디바이스 오프라인\n(우리 방식)"
    lp.font.size = Pt(16)
    lp.font.bold = True
    lp.font.color.rgb = COLOR_SUCCESS
    
    left_features = [
        "통신 불필요 (와이파이·데이터 없어도 작동)",
        "건강정보 외부 전송 안 함 (프라이버시)",
        "응답 속도 2초 이내 (로컬 처리)",
        "오래된 스마트폰도 지원 (4GB RAM)",
        "규제 리스크 낮음 (로컬 처리)"
    ]
    
    for feature in left_features:
        p = left_frame.add_paragraph()
        p.text = feature
        p.font.size = Pt(11)
        p.font.color.rgb = RGBColor(30, 30, 30)
        p.space_before = Pt(4)
        p.level = 0
    
    # 우측: 클라우드
    right_box = slide.shapes.add_shape(1, Inches(5.2), Inches(1.3), Inches(4.3), Inches(4.8))
    right_box.fill.solid()
    right_box.fill.fore_color.rgb = RGBColor(240, 240, 240)
    right_box.line.color.rgb = COLOR_NEUTRAL
    right_box.line.width = Pt(2)
    
    right_frame = right_box.text_frame
    right_frame.word_wrap = True
    right_frame.margin_left = Inches(0.2)
    right_frame.margin_top = Inches(0.2)
    
    rp = right_frame.paragraphs[0]
    rp.text = "클라우드 기반\n(참고)"
    rp.font.size = Pt(16)
    rp.font.bold = True
    rp.font.color.rgb = COLOR_NEUTRAL
    
    right_features = [
        "고성능 모델 가능 (대형 LLM)",
        "실시간 업데이트 용이",
        "서버 비용 발생",
        "네트워크 의존도 높음",
        "개인정보보호법·의료법 준수 필수"
    ]
    
    for feature in right_features:
        p = right_frame.add_paragraph()
        p.text = feature
        p.font.size = Pt(11)
        p.font.color.rgb = RGBColor(80, 80, 80)
        p.space_before = Pt(4)
        p.level = 0
    
    # 하단 결론
    conclusion_box = slide.shapes.add_shape(1, Inches(0.5), Inches(6.3), Inches(9), Inches(0.9))
    conclusion_box.fill.solid()
    conclusion_box.fill.fore_color.rgb = RGBColor(255, 245, 220)
    conclusion_box.line.color.rgb = COLOR_ACCENT
    conclusion_box.line.width = Pt(2)
    
    con_frame = conclusion_box.text_frame
    con_frame.word_wrap = True
    con_frame.vertical_anchor = MSO_ANCHOR.MIDDLE
    
    cp = con_frame.paragraphs[0]
    cp.text = "→ 시니어 의료 서비스는 프라이버시·접근성·안전성을 우선해 온디바이스 오프라인 선택"
    cp.font.size = Pt(13)
    cp.font.bold = True
    cp.font.color.rgb = COLOR_ACCENT
    cp.alignment = PP_ALIGN.CENTER

def add_pillars_slide(prs):
    """Slide 5: 3축 검증 체계"""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    background = slide.background
    fill = background.fill
    fill.solid()
    fill.fore_color.rgb = COLOR_WHITE
    
    # 제목
    title_box = slide.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(9), Inches(0.8))
    title_frame = title_box.text_frame
    tp = title_frame.paragraphs[0]
    tp.text = "[평가] 정확도 90% 달성을 위한 3축 검증 체계"
    tp.font.size = Pt(16)
    tp.font.bold = True
    tp.font.color.rgb = COLOR_PRIMARY
    title_frame.word_wrap = True
    
    # 3개 기둥
    pillars = [
        {
            "title": "① 기술 축:\n지식증류 품질",
            "items": [
                "Teacher의 의료 지식 보존도 ≥85%",
                "Student의 손실 함수 수렴",
                "온디바이스 실행 시 정확도 손실 <5%"
            ],
            "color": COLOR_PRIMARY,
            "x": 0.5
        },
        {
            "title": "② 의료 안전 축:\n위험 응답 Gate",
            "items": [
                "응급 상황 응답 정확도 100%",
                "필수 정보 포함 여부 검사",
                "금지 표현 배제 검사"
            ],
            "color": COLOR_WARNING,
            "x": 3.5
        },
        {
            "title": "③ 시니어 UX 축:\n실효성 평가",
            "items": [
                "시니어 사용자 이해도 90% 이상",
                "응답 신뢰도 90% 이상",
                "오프라인 환경 사용 성공률 95% 이상"
            ],
            "color": COLOR_SUCCESS,
            "x": 6.5
        }
    ]
    
    for pillar in pillars:
        # 기둥 박스
        pillar_shape = slide.shapes.add_shape(1, Inches(pillar["x"]), Inches(1.3), Inches(2.8), Inches(4.8))
        pillar_shape.fill.solid()
        pillar_shape.fill.fore_color.rgb = pillar["color"]
        pillar_shape.line.color.rgb = RGBColor(0, 0, 0)
        pillar_shape.line.width = Pt(1)
        
        pf = pillar_shape.text_frame
        pf.word_wrap = True
        pf.margin_left = Inches(0.15)
        pf.margin_top = Inches(0.2)
        pf.margin_right = Inches(0.15)
        
        # 제목
        tp = pf.paragraphs[0]
        tp.text = pillar["title"]
        tp.font.size = Pt(13)
        tp.font.bold = True
        tp.font.color.rgb = COLOR_WHITE
        tp.alignment = PP_ALIGN.CENTER
        
        # 항목들
        for item in pillar["items"]:
            p = pf.add_paragraph()
            p.text = "• " + item
            p.font.size = Pt(9)
            p.font.color.rgb = COLOR_WHITE
            p.space_before = Pt(6)
    
    # 하단 결론
    conclusion_box = slide.shapes.add_shape(1, Inches(0.5), Inches(6.3), Inches(9), Inches(0.9))
    conclusion_box.fill.solid()
    conclusion_box.fill.fore_color.rgb = RGBColor(0, 0, 0)
    conclusion_box.line.color.rgb = COLOR_NEUTRAL
    conclusion_box.line.width = Pt(1)
    
    con_frame = conclusion_box.text_frame
    con_frame.word_wrap = True
    con_frame.vertical_anchor = MSO_ANCHOR.MIDDLE
    
    cp = con_frame.paragraphs[0]
    cp.text = "→ 3축 모두 통과해야만 배포 가능 (가중 평균 아님)"
    cp.font.size = Pt(13)
    cp.font.bold = True
    cp.font.color.rgb = COLOR_WHITE
    cp.alignment = PP_ALIGN.CENTER

def add_summary_slide(prs):
    """Slide 6: 최종 요약"""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    background = slide.background
    fill = background.fill
    fill.solid()
    fill.fore_color.rgb = COLOR_WHITE
    
    # 제목
    title_box = slide.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(9), Inches(0.8))
    title_frame = title_box.text_frame
    tp = title_frame.paragraphs[0]
    tp.text = "[비즈니스] 공공 의료 데이터 + 13B 온디바이스 모델로 시니어 의료 접근성 혁신"
    tp.font.size = Pt(16)
    tp.font.bold = True
    tp.font.color.rgb = COLOR_PRIMARY
    title_frame.word_wrap = True
    
    sections = [
        {
            "label": "문제",
            "icon": "⚠️",
            "content": "독거·저사양·통신 불안정 환경의 시니어는 의료 정보 접근 어려움 → 응급 대응 지연",
            "color": RGBColor(255, 220, 220)
        },
        {
            "label": "해결책",
            "icon": "💡",
            "content": "공공 의료 데이터 기반 신뢰 확보 / 13B 경량 모델로 오프라인 작동 / 정확도 90% 이상 → 의료 안전 보장",
            "color": RGBColor(255, 250, 200)
        },
        {
            "label": "단계",
            "icon": "📊",
            "content": "Pilot (500 Q&A) → Beta (3,000 Q&A) → Full (10,000 Q&A) / 각 단계 GO/NO-GO 결정 지점",
            "color": RGBColor(220, 240, 255)
        },
        {
            "label": "기대효과",
            "icon": "🎯",
            "content": "시니어 의료 접근성 향상 / 의료법·개인정보보호법 준수 / 한국 공공 의료 데이터 활용 사례",
            "color": RGBColor(220, 255, 220)
        }
    ]
    
    section_height = 1.3
    start_y = 1.2
    
    for idx, section in enumerate(sections):
        y = start_y + (idx * section_height)
        
        # 배경 박스
        section_shape = slide.shapes.add_shape(1, Inches(0.5), Inches(y), Inches(9), Inches(1.1))
        section_shape.fill.solid()
        section_shape.fill.fore_color.rgb = section["color"]
        section_shape.line.color.rgb = RGBColor(200, 200, 200)
        section_shape.line.width = Pt(1)
        
        sf = section_shape.text_frame
        sf.word_wrap = True
        sf.margin_left = Inches(0.3)
        sf.margin_top = Inches(0.05)
        sf.vertical_anchor = MSO_ANCHOR.MIDDLE
        
        # 라벨 + 내용
        sp = sf.paragraphs[0]
        sp.text = section["label"] + ": " + section["content"]
        sp.font.size = Pt(11)
        sp.font.color.rgb = RGBColor(30, 30, 30)
        
        # 첫 단어 강조
        for run in sp.runs:
            if run.text.startswith(section["label"]):
                run.font.bold = True
                run.font.color.rgb = COLOR_PRIMARY
    
    # 하단 최종 결론
    final_box = slide.shapes.add_shape(1, Inches(0.5), Inches(6.2), Inches(9), Inches(0.9))
    final_box.fill.solid()
    final_box.fill.fore_color.rgb = RGBColor(255, 245, 220)
    final_box.line.color.rgb = COLOR_ACCENT
    final_box.line.width = Pt(2)
    
    ff = final_box.text_frame
    ff.word_wrap = True
    ff.vertical_anchor = MSO_ANCHOR.MIDDLE
    
    fp = ff.paragraphs[0]
    fp.text = "→ 데이터·모델·평가를 3단계로 체계화해 의료 안전 기준을 충족하면서도 현실적 배포 실현"
    fp.font.size = Pt(13)
    fp.font.bold = True
    fp.font.color.rgb = COLOR_ACCENT
    fp.alignment = PP_ALIGN.CENTER

def save_presentation(prs, filename):
    """PPT 파일 저장"""
    prs.save(filename)
    print(f"✅ PPT 파일이 저장되었습니다: {filename}")

# 메인 실행
if __name__ == "__main__":
    prs = create_presentation()
    
    # 각 슬라이드 추가
    add_title_slide(prs)           # Slide 1
    add_timeline_slide(prs)        # Slide 2
    add_table_slide(prs)           # Slide 3
    add_comparison_slide(prs)      # Slide 4
    add_pillars_slide(prs)         # Slide 5
    add_summary_slide(prs)         # Slide 6
    
    # 저장
    save_presentation(prs, "최종목표_향후계획.pptx")
    print("🎉 PPT 변환 완료!")
