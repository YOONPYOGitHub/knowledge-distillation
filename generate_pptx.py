#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
최종목표_향후계획 PPT 자동 생성 스크립트
python-pptx를 사용하여 6개 슬라이드 생성
"""

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE

# ============= 색상 정의 =============
COLOR_PRIMARY = RGBColor(0, 102, 204)      # 진 파란색
COLOR_ACCENT = RGBColor(255, 149, 0)       # 주황색
COLOR_SUCCESS = RGBColor(0, 176, 80)       # 초록색
COLOR_WARNING = RGBColor(255, 0, 0)        # 빨간색
COLOR_NEUTRAL = RGBColor(153, 153, 153)    # 회색
COLOR_WHITE = RGBColor(255, 255, 255)      # 흰색
COLOR_LIGHT_GRAY = RGBColor(242, 242, 242) # 연한 회색
COLOR_LIGHT_BLUE = RGBColor(220, 240, 255) # 연한 파란색
COLOR_LIGHT_GREEN = RGBColor(220, 250, 220) # 연한 초록색
COLOR_LIGHT_RED = RGBColor(255, 220, 220)  # 연한 빨강
COLOR_LIGHT_YELLOW = RGBColor(255, 250, 200) # 연한 노랑
COLOR_LIGHT_ORANGE = RGBColor(255, 245, 220) # 연한 주황

# ============= 기본 설정 =============
def setup_presentation():
    """프레젠테이션 기본 설정"""
    prs = Presentation()
    prs.slide_width = Inches(10)
    prs.slide_height = Inches(7.5)
    return prs

def add_text_box(slide, left, top, width, height, text, font_size=12, bold=False, 
                 color=RGBColor(0, 0, 0), alignment=PP_ALIGN.LEFT, vertical=MSO_ANCHOR.TOP):
    """텍스트 박스 추가"""
    textbox = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    text_frame = textbox.text_frame
    text_frame.word_wrap = True
    text_frame.vertical_anchor = vertical
    
    p = text_frame.paragraphs[0]
    p.text = text
    p.font.size = Pt(font_size)
    p.font.bold = bold
    p.font.color.rgb = color
    p.alignment = alignment
    
    return textbox

def add_shape_box(slide, left, top, width, height, fill_color, line_color=COLOR_WHITE, line_width=2):
    """도형 박스 추가"""
    shape = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE,
        Inches(left), Inches(top),
        Inches(width), Inches(height)
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill_color
    shape.line.color.rgb = line_color
    shape.line.width = Pt(line_width)
    return shape

# ============= Slide 1: Cover =============
def slide_1_cover(prs):
    """Slide 1: 표지"""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    
    # 배경
    background = slide.background
    fill = background.fill
    fill.solid()
    fill.fore_color.rgb = COLOR_PRIMARY
    
    # 제목
    title_box = slide.shapes.add_textbox(Inches(0.5), Inches(1.5), Inches(9), Inches(1.5))
    tf = title_box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "13B 경량 모델 + 공공 의료 데이터로\n시니어 온디바이스 헬스케어 서비스 실현"
    p.font.size = Pt(40)
    p.font.bold = True
    p.font.color.rgb = COLOR_WHITE
    p.alignment = PP_ALIGN.CENTER
    
    # 목표 박스들
    goals = [
        ("모델", "13B 지식증류\n모델\n(온디바이스 실행)", COLOR_PRIMARY),
        ("데이터", "공공의료\n데이터\n기반 의료 Q&A", COLOR_SUCCESS),
        ("정확도", "90% 이상\n(모든 의료\n상황 커버)", COLOR_ACCENT),
        ("배포", "모바일 앱 +\n웹\n(완전 오프라인)", RGBColor(128, 0, 255))
    ]
    
    box_width = 2
    for idx, (label, content, color) in enumerate(goals):
        x = 0.5 + (idx * 2.2)
        shape = add_shape_box(slide, x, 3.5, box_width, 2.5, color)
        
        tf = shape.text_frame
        tf.word_wrap = True
        tf.margin_left = Inches(0.1)
        tf.margin_top = Inches(0.1)
        tf.margin_right = Inches(0.1)
        
        p = tf.paragraphs[0]
        p.text = label
        p.font.size = Pt(14)
        p.font.bold = True
        p.font.color.rgb = COLOR_WHITE
        p.alignment = PP_ALIGN.CENTER
        
        p2 = tf.add_paragraph()
        p2.text = content
        p2.font.size = Pt(11)
        p2.font.color.rgb = COLOR_WHITE
        p2.alignment = PP_ALIGN.CENTER
        p2.space_before = Pt(6)
    
    # 하단 메시지
    add_text_box(slide, 0.5, 6.5, 9, 0.8,
                 "→ 통신 없이도 시니어가 안전하게 의료 정보에 접근하는 서비스 실현",
                 font_size=16, bold=True, color=COLOR_ACCENT, alignment=PP_ALIGN.CENTER)

# ============= Slide 2: Timeline =============
def slide_2_timeline(prs):
    """Slide 2: 타임라인"""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    background = slide.background
    fill = background.fill
    fill.solid()
    fill.fore_color.rgb = COLOR_WHITE
    
    # 제목
    add_text_box(slide, 0.5, 0.3, 9, 0.8,
                 "[방법론] 데이터 수집 → 모델 학습 → 안전 검증 → 배포의 4단계로 단계별 목표 달성",
                 font_size=18, bold=True, color=COLOR_PRIMARY)
    
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
    
    for idx, stage in enumerate(stages):
        x = 0.5 + (idx * 3)
        shape = add_shape_box(slide, x, 1.3, 2.8, 5, stage["color"], COLOR_PRIMARY, 2)
        
        tf = shape.text_frame
        tf.word_wrap = True
        tf.margin_left = Inches(0.15)
        tf.margin_top = Inches(0.15)
        
        p = tf.paragraphs[0]
        p.text = stage["name"]
        p.font.size = Pt(20)
        p.font.bold = True
        p.font.color.rgb = COLOR_PRIMARY
        p.alignment = PP_ALIGN.CENTER
        
        p = tf.add_paragraph()
        p.text = stage["period"]
        p.font.size = Pt(12)
        p.font.color.rgb = COLOR_NEUTRAL
        p.alignment = PP_ALIGN.CENTER
        
        p = tf.add_paragraph()
        p.text = "─" * 12
        p.font.size = Pt(10)
        p.font.color.rgb = COLOR_NEUTRAL
        p.alignment = PP_ALIGN.CENTER
        p.space_before = Pt(6)
        
        p = tf.add_paragraph()
        p.text = "데이터: " + stage["data"]
        p.font.size = Pt(10)
        p.font.color.rgb = RGBColor(50, 50, 50)
        p.space_before = Pt(6)
        
        p = tf.add_paragraph()
        p.text = "질의: " + stage["query"]
        p.font.size = Pt(10)
        p.font.color.rgb = RGBColor(50, 50, 50)
        p.space_before = Pt(3)
        
        p = tf.add_paragraph()
        p.text = "목표: " + stage["goal"]
        p.font.size = Pt(10)
        p.font.color.rgb = RGBColor(50, 50, 50)
        p.space_before = Pt(3)
    
    # 하단 결론
    add_text_box(slide, 0.5, 6.5, 9, 0.8,
                 "→ 각 단계마다 정확도·응답 시간·위험 응답 여부를 측정해 다음 단계 진행 여부 결정",
                 font_size=12, bold=True, color=COLOR_ACCENT, alignment=PP_ALIGN.CENTER)

# ============= Slide 3: Table =============
def slide_3_table(prs):
    """Slide 3: 테이블"""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    background = slide.background
    fill = background.fill
    fill.solid()
    fill.fore_color.rgb = COLOR_WHITE
    
    # 제목
    add_text_box(slide, 0.5, 0.3, 9, 0.8,
                 "[기술] Pilot·Beta·Full 각 단계별 데이터·모델·평가 목표 명확화",
                 font_size=16, bold=True, color=COLOR_PRIMARY)
    
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
    
    # 테이블 데이터
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
            
            if row_idx % 2 == 0:
                cell.fill.solid()
                cell.fill.fore_color.rgb = COLOR_LIGHT_GRAY
            else:
                cell.fill.solid()
                cell.fill.fore_color.rgb = COLOR_WHITE
            
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
    add_text_box(slide, 0.5, 6.2, 9, 0.8,
                 "→ 각 단계는 데이터 규모·정확도·응답 속도 목표가 명확하며, 미달 시 이전 단계로 재귀",
                 font_size=12, bold=True, color=COLOR_ACCENT, alignment=PP_ALIGN.CENTER)

# ============= Slide 4: Comparison =============
def slide_4_comparison(prs):
    """Slide 4: 비교"""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    background = slide.background
    fill = background.fill
    fill.solid()
    fill.fore_color.rgb = COLOR_WHITE
    
    # 제목
    add_text_box(slide, 0.5, 0.3, 9, 0.8,
                 "[안전] 온디바이스 오프라인 서비스 vs 클라우드 기반 서비스 비교",
                 font_size=16, bold=True, color=COLOR_PRIMARY)
    
    # 좌측: 온디바이스
    left_shape = add_shape_box(slide, 0.5, 1.3, 4.3, 4.8, COLOR_LIGHT_GREEN, COLOR_SUCCESS, 3)
    lf = left_shape.text_frame
    lf.word_wrap = True
    lf.margin_left = Inches(0.2)
    lf.margin_top = Inches(0.2)
    
    p = lf.paragraphs[0]
    p.text = "✓ 온디바이스 오프라인\n(우리 방식)"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = COLOR_SUCCESS
    p.alignment = PP_ALIGN.CENTER
    
    left_features = [
        "통신 불필요 (와이파이·데이터 없어도 작동)",
        "건강정보 외부 전송 안 함 (프라이버시)",
        "응답 속도 2초 이내 (로컬 처리)",
        "오래된 스마트폰도 지원 (4GB RAM)",
        "규제 리스크 낮음 (로컬 처리)"
    ]
    
    for feature in left_features:
        p = lf.add_paragraph()
        p.text = "• " + feature
        p.font.size = Pt(10)
        p.font.color.rgb = RGBColor(30, 30, 30)
        p.space_before = Pt(4)
    
    # 우측: 클라우드
    right_shape = add_shape_box(slide, 5.2, 1.3, 4.3, 4.8, COLOR_LIGHT_GRAY, COLOR_NEUTRAL, 2)
    rf = right_shape.text_frame
    rf.word_wrap = True
    rf.margin_left = Inches(0.2)
    rf.margin_top = Inches(0.2)
    
    p = rf.paragraphs[0]
    p.text = "클라우드 기반\n(참고)"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = COLOR_NEUTRAL
    p.alignment = PP_ALIGN.CENTER
    
    right_features = [
        "고성능 모델 가능 (대형 LLM)",
        "실시간 업데이트 용이",
        "서버 비용 발생",
        "네트워크 의존도 높음",
        "개인정보보호법·의료법 준수 필수"
    ]
    
    for feature in right_features:
        p = rf.add_paragraph()
        p.text = "• " + feature
        p.font.size = Pt(10)
        p.font.color.rgb = RGBColor(80, 80, 80)
        p.space_before = Pt(4)
    
    # 하단 결론
    conclusion = add_shape_box(slide, 0.5, 6.3, 9, 0.9, COLOR_LIGHT_ORANGE, COLOR_ACCENT, 2)
    cf = conclusion.text_frame
    cf.word_wrap = True
    cf.vertical_anchor = MSO_ANCHOR.MIDDLE
    
    p = cf.paragraphs[0]
    p.text = "→ 시니어 의료 서비스는 프라이버시·접근성·안전성을 우선해 온디바이스 오프라인 선택"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = COLOR_ACCENT
    p.alignment = PP_ALIGN.CENTER

# ============= Slide 5: Three Pillars =============
def slide_5_pillars(prs):
    """Slide 5: 3축"""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    background = slide.background
    fill = background.fill
    fill.solid()
    fill.fore_color.rgb = COLOR_WHITE
    
    # 제목
    add_text_box(slide, 0.5, 0.3, 9, 0.8,
                 "[평가] 정확도 90% 달성을 위한 3축 검증 체계",
                 font_size=16, bold=True, color=COLOR_PRIMARY)
    
    pillars = [
        {
            "title": "① 기술 축:\n지식증류 품질",
            "items": [
                "Teacher의 의료 지식\n보존도 ≥85%",
                "Student의 손실\n함수 수렴",
                "온디바이스 실행 시\n정확도 손실 <5%"
            ],
            "color": COLOR_PRIMARY,
            "x": 0.5
        },
        {
            "title": "② 의료 안전 축:\n위험 응답 Gate",
            "items": [
                "응급 상황 응답\n정확도 100%",
                "필수 정보 포함\n여부 검사",
                "금지 표현\n배제 검사"
            ],
            "color": COLOR_WARNING,
            "x": 3.5
        },
        {
            "title": "③ 시니어 UX 축:\n실효성 평가",
            "items": [
                "시니어 사용자\n이해도 90% 이상",
                "응답 신뢰도\n90% 이상",
                "오프라인 환경\n사용 성공률 95%"
            ],
            "color": COLOR_SUCCESS,
            "x": 6.5
        }
    ]
    
    for pillar in pillars:
        pillar_shape = add_shape_box(slide, pillar["x"], 1.3, 2.8, 4.8,
                                      pillar["color"], RGBColor(0, 0, 0), 1)
        pf = pillar_shape.text_frame
        pf.word_wrap = True
        pf.margin_left = Inches(0.15)
        pf.margin_top = Inches(0.2)
        pf.margin_right = Inches(0.15)
        
        tp = pf.paragraphs[0]
        tp.text = pillar["title"]
        tp.font.size = Pt(13)
        tp.font.bold = True
        tp.font.color.rgb = COLOR_WHITE
        tp.alignment = PP_ALIGN.CENTER
        
        for item in pillar["items"]:
            p = pf.add_paragraph()
            p.text = "• " + item
            p.font.size = Pt(9)
            p.font.color.rgb = COLOR_WHITE
            p.space_before = Pt(6)
    
    # 하단 결론
    conclusion = add_shape_box(slide, 0.5, 6.3, 9, 0.9, RGBColor(0, 0, 0), COLOR_NEUTRAL, 1)
    cf = conclusion.text_frame
    cf.word_wrap = True
    cf.vertical_anchor = MSO_ANCHOR.MIDDLE
    
    p = cf.paragraphs[0]
    p.text = "→ 3축 모두 통과해야만 배포 가능 (가중 평균 아님)"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = COLOR_WHITE
    p.alignment = PP_ALIGN.CENTER

# ============= Slide 6: Summary =============
def slide_6_summary(prs):
    """Slide 6: 요약"""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    background = slide.background
    fill = background.fill
    fill.solid()
    fill.fore_color.rgb = COLOR_WHITE
    
    # 제목
    add_text_box(slide, 0.5, 0.3, 9, 0.8,
                 "[비즈니스] 공공 의료 데이터 + 13B 온디바이스 모델로 시니어 의료 접근성 혁신",
                 font_size=16, bold=True, color=COLOR_PRIMARY)
    
    sections = [
        {
            "label": "문제",
            "content": "독거·저사양·통신 불안정 환경의 시니어는 의료 정보 접근 어려움 → 응급 대응 지연",
            "color": COLOR_LIGHT_RED
        },
        {
            "label": "해결책",
            "content": "공공 의료 데이터 기반 신뢰 확보 / 13B 경량 모델로 오프라인 작동 / 정확도 90% 이상",
            "color": COLOR_LIGHT_YELLOW
        },
        {
            "label": "단계",
            "content": "Pilot (500 Q&A) → Beta (3,000 Q&A) → Full (10,000 Q&A) / 각 단계 GO/NO-GO 결정",
            "color": COLOR_LIGHT_BLUE
        },
        {
            "label": "기대효과",
            "content": "시니어 의료 접근성 향상 / 의료법·개인정보보호법 준수 / 한국 공공 의료 데이터 활용",
            "color": RGBColor(220, 255, 220)
        }
    ]
    
    for idx, section in enumerate(sections):
        y = 1.2 + (idx * 1.3)
        section_shape = add_shape_box(slide, 0.5, y, 9, 1.1, section["color"], RGBColor(200, 200, 200), 1)
        sf = section_shape.text_frame
        sf.word_wrap = True
        sf.margin_left = Inches(0.3)
        sf.vertical_anchor = MSO_ANCHOR.MIDDLE
        
        p = sf.paragraphs[0]
        p.text = "● " + section["label"] + ": " + section["content"]
        p.font.size = Pt(11)
        p.font.color.rgb = RGBColor(30, 30, 30)
    
    # 하단 최종 결론
    final = add_shape_box(slide, 0.5, 6.2, 9, 0.9, COLOR_LIGHT_ORANGE, COLOR_ACCENT, 2)
    ff = final.text_frame
    ff.word_wrap = True
    ff.vertical_anchor = MSO_ANCHOR.MIDDLE
    
    p = ff.paragraphs[0]
    p.text = "→ 데이터·모델·평가를 3단계로 체계화해 의료 안전 기준을 충족하면서도 현실적 배포 실현"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = COLOR_ACCENT
    p.alignment = PP_ALIGN.CENTER

# ============= Main =============
def main():
    """메인 실행 함수"""
    prs = setup_presentation()
    
    print("🎬 PPT 슬라이드 생성 중...")
    
    slide_1_cover(prs)
    print("✅ Slide 1: Cover")
    
    slide_2_timeline(prs)
    print("✅ Slide 2: Timeline (Pilot→Beta→Full)")
    
    slide_3_table(prs)
    print("✅ Slide 3: Table (데이터·모델·평가)")
    
    slide_4_comparison(prs)
    print("✅ Slide 4: Comparison (온디바이스 vs 클라우드)")
    
    slide_5_pillars(prs)
    print("✅ Slide 5: Three Pillars (3축 검증)")
    
    slide_6_summary(prs)
    print("✅ Slide 6: Summary (최종 요약)")
    
    # 저장
    output_file = "최종목표_향후계획.pptx"
    prs.save(output_file)
    print(f"\n🎉 PPT 파일 저장 완료: {output_file}")

if __name__ == "__main__":
    main()
