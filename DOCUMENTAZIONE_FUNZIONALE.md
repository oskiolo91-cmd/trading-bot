# Documento funzionale — Trading Bot

**Stato del documento:** 2 ottobre 2026  
**Ambito:** applicazione Streamlit, esecuzione ordini Alpaca, screener, backtest e persistenza locale presenti nel repository.

Questo documento descrive il comportamento rilevabile nel codice. Non costituisce una raccomandazione finanziaria né una garanzia di esecuzione o rendimento. Le protezioni lato broker riducono alcuni rischi operativi, ma non eliminano gap, slippage, rifiuti, indisponibilità API o perdita di capitale.

## 1. Sintesi del sistema

Il repository comprende quattro aree principali:

1. **Dashboard Streamlit** per osservazione dei ticker, segnali, posizioni, impostazioni di rischio, ordini manuali e gestione dei bot.
2. **Esecuzione live Alpaca Paper Trading** con ordini limit di acquisto e trailing stop nativi associati ai fill.
3. **Market radar / screener** che arricchisce un catalogo di titoli con indicatori, metadati e validatori di strategia.
4. **Backtest offline** su barre giornaliere, con report delle operazioni e metriche.

La dashboard e il loop autonomo in `live_trader.py` sono due modalità distinte di esecuzione. La dashboard dipende dalla pagina Streamlit e dal suo ciclo di refresh; il loop autonomo riceve barre Alpaca via WebSocket. Entrambi operano in modalità paper tramite il client configurato nel codice.

## 2. Componenti e responsabilità

| Componente | Responsabilità |
|---|---|
| `dashboard.py` | UI Streamlit, selezione ticker, presentazione di prezzi/indicatori/posizioni, avvio dei cicli bot, ordini manuali, aggiornamento delle impostazioni di rischio |
| `live_trader.py` | Client Alpaca, submission/cancellazione ordini, stream fill, gestione ordini trailing, ledgers per ticker, loop live autonomo |
| `bot_state.py` | Lettura/scrittura atomica di impostazioni per simbolo in JSON |
| `alpaca_data.py` | Richieste Alpaca di barre giornaliere RAW per uno o più simboli |
| `indicators.py` | Calcolo pandas di Bande di Bollinger, RSI, ADX e ATR |
| `signals.py` | Regole pure di segnale/uscita utilizzate principalmente dal backtest |
| `pnl_manager.py` | Ricostruzione FIFO dei fill per quantità e P&L realizzato |
| `backtest.py` | Simulazione giornaliera e walk-forward |
| `screener.py` | Costruzione e aggiornamento del market radar CSV |
| `macro_filter.py` | Filtro da calendario JSON di eventi con impatto configurato |
| `main.py` | CLI per walk-forward backtest con input CSV oppure dati sintetici |

## 3. Dashboard: funzionalità e comportamento

### 3.1 Connessione e conto

- Legge `ALPACA_API_KEY` e `ALPACA_SECRET_KEY` dall'ambiente o dai Secrets Streamlit.
- Se le chiavi mancano, mostra la watchlist ma disabilita le funzioni di trading.
- Legge equity del conto e posizioni aperte da Alpaca.
- Il client trading è impostato per **Paper Trading**.
- Usa `ALPACA_DATA_FEED` (`iex` predefinito, oppure `sip`) per i dati storici.

### 3.2 Ticker, dati e grafici

- Recupera asset azionari USA attivi, negoziabili e fractional-enabled per popolare la selezione.
- Consente di conservare nella selezione ticker con bot attivo o posizione aperta anche se non compaiono nei filtri correnti.
- Scarica barre giornaliere Alpaca, calcola indicatori e aggiorna i dati in background; il refresh standard è 60 secondi.
- Mostra prezzo, variazione, ADX, RSI, segnale, candlestick daily, Bollinger Bands, volume e, in modalità trend, SMA 200.
- Nel grafico visualizza gli acquisti eseguiti e gli ordini sell/stop aperti rilevabili dal broker.
- Include radar con ricerca, filtri per metadati e validator, paginazione e attivazione ticker.

**Segnale di ingresso mostrato in dashboard:** `ADX < soglia`, `RSI < soglia` e prezzo `<= Bollinger inferiore`.

### 3.3 Profili di rischio

Valori attualmente definiti in dashboard:

| Profilo | ADX max | RSI max | Budget indicativo | Trailing stop |
|---|---:|---:|---:|---:|
| Conservativo | 20 | 30 | $50 | 3% |
| Bilanciato (predefinito) | 25 | 35 | $100 | 6% |
| Speculativo | 35 | 45 | $200 | 12% |
| Custom | Configurabile | Configurabile | Configurabile | Configurabile |

Nel profilo Custom sono presenti controlli per budget, soglie ADX/RSI, moltiplicatori ATR, trailing, obiettivo giornaliero e limite di perdita giornaliero.

È possibile anche modificare la percentuale trailing per ticker dalla tabella personale. Un cambio profilo o trailing salva le impostazioni e tenta di sostituire (`replace`) gli ordini trailing nativi aperti per quel simbolo; gli errori sono riportati in log e UI.

### 3.4 Acquisti e protezione broker

- Il bot acquista con ordini **limit DAY**.
- Le quantità degli ordini configurati con trailing nativo vengono arrotondate per difetto ad azioni intere.
- Un budget insufficiente per comprare almeno un'azione intera non produce un ordine.
- Dopo un evento di fill o partial fill, il gestore degli aggiornamenti invia un ordine sell `trailing_stop` GTC con `trail_percent` del profilo.
- Il ciclo live autonomo verifica inoltre periodicamente l'esistenza/quantità della protezione mentre la posizione è aperta.
- La dashboard mostra il `stop_price` effettivo dell'ordine trailing restituito da Alpaca, se disponibile.
- In caso di stop legacy già aperti, la riconciliazione si ferma senza cancellarli automaticamente.

Il trailing non viene inviato in modo atomico insieme al buy: l'ordine di protezione è creato al fill. Tra il fill e l'accettazione del trailing può quindi esistere un intervallo non protetto.

### 3.5 Vendite, kill e limiti giornalieri

- Le vendite manuali annullano gli ordini sell aperti prima di inviare un market sell.
- Il kill simbolo annulla gli ordini del simbolo, chiude la posizione e disattiva il bot, se le richieste Alpaca riescono.
- La logica del bot consulta il P&L realizzato giornaliero FIFO per simbolo. Al raggiungimento dell'obiettivo o del limite negativo chiude la posizione/ordini del simbolo e interrompe le ulteriori operazioni per il giorno.
- Il loop standalone `DAILY_SCALPER` include un blocco di ingresso/chiusura negli ultimi 15 minuti della sessione; non è un'opzione selezionabile dalla dashboard.

### 3.6 Metriche del conto

- **Account Equity** e valore delle posizioni provengono da Alpaca.
- Il KPI P&L è calcolato come somma del P&L realizzato ricostruito dai fill e del P&L non realizzato riportato sulle posizioni aperte.
- Il grafico storico di equity del conto è una serie separata: può riflettere depositi/prelievi e non va interpretato come curva pura della strategia.
- Il P&L giornaliero mostrato come intraday usa il valore intraday fornito da Alpaca, con fallback al P&L non realizzato totale.

## 4. Persistenza dello stato

`bot_state.py` usa `bot_state.json` nella cartella del progetto; il percorso può essere sovrascritto con `BOT_STATE_PATH`. Il file è ignorato da Git.

Per ogni simbolo vengono salvati i dati disponibili tra:

- nome profilo e trailing percent effettiva;
- trailing personalizzato, se impostato;
- impostazioni Custom;
- high-water mark locale;
- data dell'ultimo acquisto del giorno.

La scrittura avviene in un file temporaneo, con flush/fsync e sostituzione atomica. Un file mancante avvia uno stato vuoto. JSON corrotto o struttura non valida genera log e fallback vuoto. I parametri UI vengono caricati in `st.session_state` all'avvio.

Se manca l'HWM salvato e viene ricostruita una posizione, il codice cerca i massimi giornalieri Alpaca dalla data di ingresso; in assenza di dati validi usa il prezzo medio di carico e registra l'errore.

**Importante:** l'HWM locale non determina il trailing stop attivo presso Alpaca. Il broker gestisce autonomamente il massimo osservato dal proprio ordine trailing; il dato locale è di recovery/visualizzazione e non ripristina il massimo interno del broker dopo la cancellazione o sostituzione dell'ordine.

## 5. Market radar e screener

- Il radar usa un catalogo derivato da componenti S&P 500 e Nasdaq 100 o da simboli forniti manualmente.
- I metadati Nasdaq disponibili includono nome, settore, industria e market cap.
- I dati di mercato e gli indicatori sono ottenuti da Alpaca e persistiti in `market_radar.csv`; il master locale è `nasdaq_screener.csv`.
- L'arricchimento viene eseguito in batch e aggiorna il CSV in modo incrementale.
- Il radar calcola:
  - volume medio a 20 giorni;
  - ADX, ATR percentuale, RSI, Bollinger inferiore e SMA 200;
  - validator scalper: ADX < 25, RSI < 35, close <= Bollinger inferiore;
  - validator trend: condizioni scalper più close > SMA 200.
- `screen_assets` supporta filtri di volume, ADX, rapporto laterale, ATR minimo e calendario macro.
- Il calendario macro è un file JSON fornito dall'utente; gli eventi ad alto impatto possono escludere i giorni selezionati.

Le barre daily della giornata corrente sono escluse dal radar quando il mercato risulta aperto o quando la lettura dell'orologio Alpaca fallisce.

## 6. Backtest e CLI

- Indicatori e simulazione lavorano su barre giornaliere.
- Sono supportati CSV OHLCV oppure dati sintetici generati deterministicamente dalla CLI.
- Il backtest simula ordini limit, commissioni configurabili, quantità fractional, stop ATR e uscite su candele tramite regole in `signals.py`.
- Il walk-forward divide i dati in porzione in-sample e out-of-sample; il default CLI è 70%/30%.
- Metriche: P&L netto, win rate, massimo drawdown, numero trade, profit factor, capitale finale e rendimento.
- I risultati CLI vengono salvati come `trades.csv` e `metrics.json`.
- La dashboard esegue un confronto dei profili predefiniti su dati giornalieri recenti.

Il backtest **non replica l'esecuzione nativa Alpaca**: è una simulazione a barre, non modella tick, latenza, fill parziali reali, slippage effettivo o l'ordine trailing broker. Le sue metriche non sono direttamente confrontabili con il comportamento live.

## 7. Specifiche tecniche e parametri operativi

### Variabili d'ambiente

| Variabile | Uso | Default/valori |
|---|---|---|
| `ALPACA_API_KEY` | Credenziale Alpaca | Obbligatoria per trading e market data autenticati |
| `ALPACA_SECRET_KEY` | Segreto Alpaca | Obbligatoria per trading e market data autenticati |
| `ALPACA_DATA_FEED` | Feed dati | `iex` predefinito; `sip` accettato |
| `BOT_STATE_PATH` | Percorso JSON persistente | `bot_state.json` accanto ai moduli |
| `BOT_MODE` | Modalità del loop autonomo | `TREND_FOLLOWER` predefinita; `DAILY_SCALPER` alternativa |
| `TRADING_SYMBOL` | Simbolo del loop standalone | `SPY` predefinito |

### Dipendenze principali

Python con `alpaca-py`, Streamlit, pandas, numpy e Plotly; elenco in `requirements.txt`.

### Ordini e unità

- `StrategyParams.trailing_pct` è una frazione (es. `0.06`).
- La richiesta Alpaca riceve `trail_percent` espresso in punti percentuali (es. `6.0`).
- Gli ordini d'ingresso hanno TIF DAY.
- I trailing stop sono GTC.
- Gli ordini trailing nativi vengono inviati soltanto per quantità intere secondo la logica attuale.

## 8. Criticità e miglioramenti raccomandati

Priorità indicative: **P0** rischio di esposizione/protezione; **P1** correttezza/coerenza funzionale; **P2** resilienza e manutenibilità.

| Priorità | Area | Criticità osservata | Miglioramento raccomandato |
|---|---|---|---|
| P0 | Finestra buy/fill/stop | Buy e trailing sono due richieste distinte. Un fill può precedere l'accettazione del trailing; errore di rete o rifiuto lascia la posizione senza quella protezione. | Introdurre una macchina a stati per gli ingressi: fill parziale, quantità protetta, retry con backoff, allarme persistente e blocco di nuovi acquisti finché la copertura non è confermata. Valutare con Alpaca le protezioni atomiche supportate per il prodotto e la quantità in uso. |
| P0 | Posizioni esistenti | Posizioni frazionarie e ordini sell legacy non vengono convertiti automaticamente. La riconciliazione registra l'errore ma può lasciare una posizione senza trailing nativo. | Dashboard di stato “protetto/non protetto”, alert bloccante e procedura esplicita di migrazione/manuale per ogni posizione residua. |
| P0 | Limite di perdita | Il limite giornaliero usa P&L realizzato; una perdita non realizzata intraday non lo attiva. Il trailing broker resta la protezione principale, ma può avere gap/slippage. | Definire e implementare un limite account/simbolo che includa realized + unrealized, con policy di chiusura e gestione delle richieste rifiutate. |
| P0 | Riconciliazione quantità | La protezione è basata sulla quantità totale in posizione per simbolo, non su una distinzione certa tra lotti generati dal bot e lotti manuali. Ordini/posizioni manuali sullo stesso ticker possono interferire. | Associare ordini a client-order-id/strategia e gestire quantità posseduta dal bot in modo verificabile; impedire sovrapposizioni con trading manuale. |
| P1 | Dimensionamento | Per supportare trailing nativo il sistema compra azioni intere. Con i budget attuali molti ticker dal prezzo elevato non possono generare ordini; parte del budget resta inutilizzata. | Esporre chiaramente “quantità zero/budget insufficiente” e consentire budget per ticker o una protezione broker compatibile con fractional shares, dopo verifica API. |
| P1 | Configurazione ATR/take-profit | I controlli `stop_loss_atr_mult` e `take_profit_atr_mult` sono ancora mostrati/configurabili, ma gli ordini live attuali non inviano lo stop ATR né il take-profit. Alcune strutture del backtest conservano invece i parametri precedenti. | Rimuovere i controlli non applicati o reintrodurre esplicitamente ordini/protezioni compatibili; separare la configurazione live da quella backtest e aggiornare le etichette UI. |
| P1 | Strategia dashboard vs standalone | `compute_signal` in dashboard valuta ADX/RSI/Bollinger; la modalità `TREND_FOLLOWER` del controllo standalone applica anche SMA 200. Le due modalità non hanno quindi la stessa regola d'ingresso. | Centralizzare la valutazione dei segnali e usare la stessa implementazione/parametri su dashboard, screener, standalone e backtest, oppure documentare chiaramente le differenze. |
| P1 | Configurazione strategia | La dashboard espone profili di rischio ma non una selezione `TREND_FOLLOWER`/`DAILY_SCALPER`; alcune soglie di modello restano visibili senza essere usate negli ingressi dashboard. | Separare profilo rischio da modalità strategia e mostrare nell'interfaccia la strategia realmente in esecuzione. |
| P1 | Vendita manuale | La vendita annulla prima gli ordini protettivi. Se il market sell fallisce dopo l'annullamento, la posizione può restare aperta senza stop fino a una successiva riconciliazione. | Rendere la vendita un workflow transazionale/recuperabile: verificare stato broker, ripristinare protezione in caso di fallimento e segnalare la posizione scoperta. |
| P1 | Modifica trailing | `replace` sostituisce la percentuale, ma non ripristina il massimo interno precedente del trailing; il nuovo livello effettivo può comportarsi diversamente dal vecchio ordine. | Verificare la semantica Alpaca di replace per il trailing; visualizzare il livello aggiornato dopo conferma broker e segnalare sostituzioni pendenti/rifiutate. |
| P1 | KPI rendimento | La somma P&L realized + unrealized evita di trattare un deposito come profitto, ma la percentuale usa `base_value` del conto: depositi/prelievi e flussi successivi possono rendere il denominatore non rappresentativo del capitale strategia. | Calcolare time-weighted return o money-weighted return usando flussi di cassa effettivi; fornire P&L del bot separato dal P&L totale del conto. |
| P1 | Commissioni P&L | Il ledger FIFO stima le commissioni con `commission_pct`; non è necessariamente uguale alle fee effettive Alpaca. Fill non abbinati non vengono conteggiati come P&L realizzato. | Usare fee/activity broker se disponibili, esporre righe non abbinate come errore e aggiungere riconciliazione ledger-versus-conto. |
| P1 | Backtest vs live | Backtest giornaliero applica stop/uscite su OHLC e fractional quantities; live usa ordini nativi e quantità intere. La dashboard confronta risultati che non descrivono la stessa esecuzione. | Allineare sizing e lifecycle alle regole live o etichettare il backtest come modello teorico non equivalente; includere slippage e gap scenario. |
| P1 | HWM/recovery | Il recovery usa high giornalieri e quindi non ricostruisce il massimo intraday esatto. Non è lo stesso HWM conservato dal broker nel trailing order. | Trattare l'HWM broker come autoritativo quando recuperabile; segnalare esplicitamente che il valore locale è una stima daily. |
| P2 | Persistenza locale | JSON locale è atomico entro un processo, ma non è un database transazionale condiviso tra più istanze; lock Python non coordina processi diversi. In ambienti effimeri il disco può essere perso. | Per deploy multiistanza usare storage durevole e con lock/versioning (DB o servizio state store); backup, schema/versioni e migrazioni. |
| P2 | Gestione JSON corrotto | Il loader segnala il problema e riparte con stato vuoto; il file guasto non viene automaticamente archiviato e impostazioni precedenti possono andare perse. | Creare copia `.corrupt` con timestamp prima del fallback, validare schema per campo e fornire procedura di recovery. |
| P2 | Streamlit | Il ciclo bot della dashboard gira mentre la pagina è attiva/aggiornata; non è un worker sempre attivo garantito. | Separare l'esecuzione in un servizio persistente con supervisione, heartbeat e dashboard come client di sola gestione/monitoraggio. |
| P2 | Dati radar | Universo, metadati esterni e disponibilità bars dipendono da fonti esterne (Wikipedia/Nasdaq/Alpaca), rate limit e aggiornamento manuale/asincrono. | Rendere fonti e timestamp visibili, monitorare staleness, versionare il catalogo e gestire retry/rate limit. |
| P2 | Osservabilità | I log bot sono principalmente in memoria Streamlit e limitati; parte degli errori è soltanto registrata e non genera un allarme persistente. | Log strutturato persistente, identificatori ordine/strategia, metriche di copertura stop e notifiche per ordini respinti o posizioni scoperte. |

## 9. Verifiche eseguite e non eseguite

- La suite presente nel repository è stata eseguita: **57 test superati** nell'ultima verifica documentata.
- I test coprono funzioni pure, simulazioni/fake client, persistenza JSON, sostituzione trailing, recovery HWM, logica last-buy e dashboard.
- Non risultano verifiche end-to-end con ordini reali o paper inviati a Alpaca durante la preparazione di questo documento.
- I test unitari non provano la disponibilità continuativa del websocket, i tempi di accettazione Alpaca, il comportamento in gap o la persistenza in un deploy cloud.

## 10. Criteri proposti per la prossima fase

1. Rendere verificabile e osservabile la copertura stop di ogni posizione, incluse quelle frazionarie e legacy.
2. Gestire fill parziali e errori con retry controllato senza duplicare quantità o lasciare posizioni scoperte.
3. Definire un limite di rischio intraday che includa P&L non realizzato e comportamento su errore broker.
4. Allineare le regole di segnale e sizing tra dashboard, standalone, screener e backtest.
5. Rimuovere o riattivare coerentemente i parametri ATR/take-profit esposti nella UI.
6. Migrare lo stato locale a storage durevole se il bot deve operare in più istanze o senza browser aperto.
7. Calcolare metriche di performance della strategia indipendenti dai flussi di cassa e dalle fee stimate.
