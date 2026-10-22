"""Captura Inicio y Exploración reales de Streamlit y genera PDF de sus vistas.
Dependencias opcionales: playwright, reportlab, pillow. No altera app ni datos.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time
import urllib.request

from PIL import Image
from playwright.sync_api import sync_playwright
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'docs'/'pdf'
CAPTURES=OUTPUT/'capturas'
PORT=8531
manifest={'fecha':'2026-10-22','version_datos':'Datos tal como se recibieron',
          'filtros':'Sin filtros: todos los contratos e internet; incluir contratos ausentes',
          'fuentes':[], 'vistas':[]}
for p in sorted((ROOT/'data').glob('telco_csv_*.csv')):
    manifest['fuentes'].append({'archivo':p.name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})


class Capturer:
    def __init__(self,page):
        self.page=page
        self.counter=0
        self.pages={'Inicio':[],'Exploracion':[]}
    def save(self,section,label,locator=None):
        self.counter+=1
        path=CAPTURES/f'{self.counter:03d}_{section}.png'
        if locator is None:
            main=self.page.locator('[data-testid="stMain"]')
            bounds=main.bounding_box()
            self.page.screenshot(path=str(path),clip={'x':bounds['x'],'y':60,
                'width':bounds['width']-2,'height':bounds['height']-62},animations='disabled')
        else:
            locator.scroll_into_view_if_needed()
            self.page.wait_for_timeout(180)
            locator.screenshot(path=str(path),animations='disabled')
        self.pages[section].append((label,path))
        manifest['vistas'].append({'seccion':section,'titulo':label,'captura':str(path.relative_to(ROOT))})
    def view(self,section,label):
        main=self.page.locator('[data-testid="stMain"]')
        main.evaluate('(e)=>e.scrollTop=0')
        self.page.wait_for_timeout(250)
        sizes=main.evaluate('(e)=>({height:e.clientHeight,full:e.scrollHeight})')
        maximum=max(0,sizes['full']-sizes['height'])
        count=max(1,math.ceil(maximum/500))
        positions=[maximum*i/count for i in range(count+1)] if maximum>5 else [0]
        for index,pos in enumerate(dict.fromkeys(positions)):
            main.evaluate('(e,y)=>e.scrollTop=y',pos)
            self.page.wait_for_timeout(250)
            self.save(section,f'{label} · tramo {index+1}')
        print(f'Capturada: {label}',flush=True)
    def grid(self,section,label,grid):
        grid.scroll_into_view_if_needed()
        scroller=grid.locator('.dvn-scroller').first
        sizes=scroller.evaluate('(e)=>({w:e.clientWidth,h:e.clientHeight,sw:e.scrollWidth,sh:e.scrollHeight})')
        def positions(full,visible):
            maximum=max(0,full-visible)
            if maximum<8: return [0]
            count=max(1,math.ceil(maximum/max(1,visible-60)))
            return [round(maximum*i/count) for i in range(count+1)]
        xs=positions(sizes['sw'],sizes['w'])
        ys=positions(sizes['sh'],sizes['h'])
        for yi,y in enumerate(dict.fromkeys(ys)):
            for xi,x in enumerate(dict.fromkeys(xs)):
                scroller.evaluate('(e,p)=>{e.scrollLeft=p[0];e.scrollTop=p[1]}',[x,y])
                self.page.wait_for_timeout(180)
                self.save(section,f'{label} · filas {yi+1}, columnas {xi+1}',grid)
        scroller.evaluate('(e)=>{e.scrollTop=0;e.scrollLeft=0}')
    def pdf(self,section):
        path=OUTPUT/f'{section}.pdf'
        c=canvas.Canvas(str(path),pageCompression=1)
        c.setTitle(f'Telco · {section} · Capturas de Streamlit')
        c.setAuthor('Natalia Gutiérrez Soliz')
        for index,(label,image) in enumerate(self.pages[section],start=1):
            with Image.open(image) as im:
                w,h=im.size
            width=842
            scale=(width-48)/w
            height=max(180,h*scale+83)
            c.setPageSize((width,height))
            c.setFillColorRGB(.08,.20,.26)
            c.setFont('DejaVu',11)
            c.drawString(24,height-23,label[:125])
            c.setFont('DejaVu',7)
            state='Inicio' if section=='Inicio' else 'Datos tal como se recibieron · Sin filtros'
            c.drawString(24,height-38,f'Telco · {state} · 22/10/2026')
            c.drawImage(ImageReader(str(image)),24,34,width=w*scale,height=h*scale)
            c.setFont('DejaVu',7)
            c.drawString(24,17,'Captura real de la aplicación · Fuentes y estados registrados en manifiesto_exportacion.json')
            c.drawRightString(width-24,17,f'{index} / {len(self.pages[section])}')
            c.showPage()
        c.save()
        print(f'PDF: {path} ({len(self.pages[section])} páginas)',flush=True)


def wait(page):
    page.wait_for_timeout(650)
    errors=page.locator('[data-testid="stException"]')
    if errors.count():
        raise RuntimeError(errors.first.inner_text())


def expand(page,title):
    summary=page.locator('summary').filter(has_text=re.compile(re.escape(title))).first
    if not summary.locator('..').evaluate('(e)=>e.open'):
        summary.click();page.wait_for_timeout(300)


def collapse(page,title):
    summary=page.locator('summary').filter(has_text=re.compile(re.escape(title))).first
    if summary.locator('..').evaluate('(e)=>e.open'):
        summary.click();page.wait_for_timeout(200)


def select(page,label,value):
    widget=page.locator('[data-testid="stSelectbox"]').filter(has=page.get_by_text(label,exact=True))
    widget.get_by_role('combobox').click()
    option=page.get_by_role('option',name=value,exact=True)
    if not option.count():
        page.keyboard.insert_text(value)
    option.click()
    wait(page)


def visible_grids(capture,section,label,scope=None):
    scope=scope or capture.page
    grids=scope.locator('[data-testid="stDataFrame"]')
    for i in range(grids.count()):
        grid=grids.nth(i)
        if grid.is_visible():
            capture.grid(section,f'{label} · tabla {i+1}',grid)


def main():
    OUTPUT.mkdir(parents=True,exist_ok=True);CAPTURES.mkdir(parents=True,exist_ok=True)
    pdfmetrics.registerFont(TTFont('DejaVu','/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'))
    log=open('/tmp/telco-export-pdf.log','w')
    server=subprocess.Popen([str(ROOT/'.venv/bin/python'),'-m','streamlit','run',str(ROOT/'app.py'),
        '--server.headless=true','--server.address=127.0.0.1',f'--server.port={PORT}'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    try:
        for _ in range(60):
            if server.poll() is not None: raise RuntimeError(Path('/tmp/telco-export-pdf.log').read_text())
            try:
                urllib.request.urlopen(f'http://127.0.0.1:{PORT}/_stcore/health',timeout=1);break
            except OSError: time.sleep(.5)
        with sync_playwright() as pw:
            candidates=list((Path.home()/'.cache/ms-playwright').glob('chromium-*/chrome-linux64/chrome'))
            options={'headless':True,'args':['--no-sandbox']}
            if candidates: options['executable_path']=str(sorted(candidates)[-1])
            browser=pw.chromium.launch(**options)
            page=browser.new_page(viewport={'width':1440,'height':1000},device_scale_factor=1.5)
            page.goto(f'http://127.0.0.1:{PORT}')
            page.get_by_text('Objetivo general',exact=True).wait_for(timeout=60000)
            wait(page)
            captures=Capturer(page)
            expand(page,'Detalles de la integración')
            captures.view('Inicio','Inicio · objetivo, archivos, integración y diccionario')
            visible_grids(captures,'Inicio','Integración completa')
            for filename in ['telco_csv_1.csv','telco_csv_2.csv','telco_csv_3.csv']:
                select(page,'Ver datos de',filename)
                captures.grid('Inicio',f'Vista previa: {filename} · primeras 30 filas',page.locator('[data-testid="stDataFrame"]').nth(1))
            captures.pdf('Inicio')
            page.get_by_text('Exploración',exact=True).first.click()
            page.get_by_text('Estadísticas descriptivas de las variables numéricas',exact=True).wait_for(timeout=60000)
            wait(page)
            for label in ['Resumen numérico','Distribuciones numéricas','Categorías','Abandono por contrato']:
                page.get_by_role('tab',name=label,exact=True).click();wait(page)
                captures.view('Exploracion',f'Exploración · {label}')
                if label in ['Resumen numérico','Abandono por contrato']:
                    visible_grids(captures,'Exploracion',label,page.get_by_role('tabpanel').filter(visible=True))
            page.get_by_role('tab',name='Categorías',exact=True).click();wait(page)
            from src.exploration import LABELS
            from src.data_processing import CATEGORIES
            for col in list(CATEGORIES)+['Churn']:
                select(page,'Variable',LABELS[col])
                panel=page.get_by_role('tabpanel').filter(visible=True)
                captures.grid('Exploracion',f'Frecuencias · {LABELS[col]}',panel.locator('[data-testid="stDataFrame"]').first)
            page.get_by_role('tab',name='Resumen numérico',exact=True).click();wait(page)
            expand(page,'Calidad de datos')
            expand(page,'Detalle por columna y coherencia entre servicios')
            expand(page,'Clientes con campos diferentes después de normalizar contratos')
            captures.view('Exploracion','Calidad de datos · controles y conflictos')
            quality=page.locator('summary').filter(has_text=re.compile('^.*Calidad de datos')).first.locator('..')
            visible_grids(captures,'Exploracion','Calidad de datos',quality)
            collapse(page,'Calidad de datos')
            expand(page,'KPIs complementarios del trabajo académico')
            for grouping in ['Servicio de internet','Intervalo de antigüedad','Intervalo de cargo mensual']:
                select(page,'Comparar abandono por',grouping)
                kpis=page.locator('summary').filter(has_text='KPIs complementarios').first.locator('..')
                kpis.scroll_into_view_if_needed();wait(page)
                captures.save('Exploracion',f'KPIs complementarios · {grouping}',kpis)
                visible_grids(captures,'Exploracion',f'KPIs · {grouping}',kpis)
            captures.pdf('Exploracion')
            browser.close()
        manifest['pdfs']=[{'archivo':str((OUTPUT/f'{name}.pdf').relative_to(ROOT)),
                           'paginas':len(captures.pages[name])} for name in ['Inicio','Exploracion']]
        manifest['version_app_sha256']=hashlib.sha256((ROOT/'app.py').read_bytes()).hexdigest()
        (OUTPUT/'manifiesto_exportacion.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    finally:
        server.terminate();server.wait(timeout=15);log.close()


if __name__=='__main__':
    import sys
    sys.path.insert(0,str(ROOT))
    main()
