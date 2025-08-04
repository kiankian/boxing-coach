This is me boxing with a heavy bag.
Analyze the video at 1 fps.
Create one entry per bag impact (or clear miss/glance).

Each entry should include:
- timestamp_of_outcome — (M:SS.s)
- result — "landed", "glancing", or "missed"
- punch_type — "jab", "cross", "lead hook", "rear hook", "lead uppercut", "rear uppercut", "overhand", "body jab", "body hook", etc.
- bag_zone — "high", "mid", or "low"
- feedback — form, power mechanics, range, wrist alignment, recovery. Only if needed, max one every 3 seconds - otherwise give an empty string
- punch_quality — "good" or "bad" based on overall technique and impact

Track running totals (include after feedback):
- total_good_punches
- total_bad_punches

Output ONLY the raw JSON object under the top-level key "punches". No code fences, no markdown, no extra commentary.