# Arxitektura

```
Foydalanuvchi
   |  (Streamlit dashboard)         (REST klient)
   v                                      v
app/ui/dashboard.py              app/api/server.py (FastAPI)
        \                              /
         +--------> app/agents.run_agent <--------+
                          |
                 LangGraph (graph.py)
        planner --> executor (sikl) --> responder
           |            |                  |
      LLM / qoidalar   app/tools        LLM / shablon
                        registry
              +----------+-----------+-------------+
              v          v           v             v
        app/analysis  app/ml   app/visualization  tools/report
                          \         |            /
                           v        v           v
                   app/services (storage, datasets)
                           |
                   app/database (SQLite: sessions, conversations,
                                 datasets, models, charts)
```

## Asosiy qarorlar

- **LLM ixtiyoriy.** `planner.make_plan` avval LLM'dan JSON reja so'raydi; xato yoki bo'sh javobda `heuristic_plan` ishlaydi. Natija matni uchun ham shunday: tayyor xulosa (`summary.py`) har doim mavjud.
- **Vositalar xatoni qaytaradi, tashlamaydi.** `run_tool` har doim `{"success": bool, ...}` lug'ati beradi; agent bitta qadam xato bersa ham qolganini bajaradi.
- **Raqamlar LLM'dan emas, kod hisoblaydi.** LLM faqat tayyor natijani tushuntiradi.
- **Sessiyalar** barcha ma'lumotni (dataset, model, grafik, tarix) bog'laydi; sessiya o'chirilsa, bog'liq yozuvlar ham o'chadi.

## Yangi vosita qo'shish

1. `app/tools/registry.py` da `tool_nomi(ctx: ToolContext, **args) -> dict` funksiyasini yozing.
2. Uni `TOOLS` va `TOOL_DESCRIPTIONS` ga qo'shing.
3. `app/agents/planner.py` da kalit so'zlarni, `app/agents/summary.py` da xulosa shablonini qo'shing.
4. `tests/test_tools.py` ga test yozing.
