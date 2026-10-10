# Setup help plan (organizer notes, not for participants)

One person runs the workshop, so the plan keeps live troubleshooting short and moves problems out of the session quickly. Do not put participant names or emails in this file or anywhere in the repository.

## Before the workshop (Oct 13 to Oct 19)

1. **Inbox:** quantitative.sustainable.design@gmail.com receives replies to the setup email. Target: answer within 24 hours.
2. **Tracking:** keep a private list outside the repository, with three states per person: READY (replied with the READY line), PROBLEM (sent output, waiting on a fix), SILENT (no reply).
3. **Standard replies:** most problems map to a row of the common-errors table in `setup/INSTALL.md`. Reply with the matching fix and ask for the new full output of `check_environment.py`.
4. **Fallback offer:** after one failed attempt, tell the person to use the Binder or Colab link (both tested) and to reply when the check says READY there. Do not spend more than two email rounds per person before this.
5. **Oct 16 deadline:** on Oct 17, send a short nudge to SILENT people with the same instructions (use the setup email text, shortened). On Oct 19, send the session 1 reminder (`communication/02_reminder_session1.md`).
6. **Known failure modes to expect** (all in the common-errors table): Python below 3.12, Graphviz program missing, Windows long paths, Spyder or VS Code pointing at a different Python, numpy mismatch in Colab until the session is restarted.

## Session 1 (Oct 20)

| Time (ET) | Action |
|---|---|
| 11:45 | Open the Zoom room (enable "allow participants to join before host", or start the meeting early). Not recorded. Anyone with a problem can ask in the chat or talk directly. |
| 12:00 | Start recording and welcome. Show the READY line and ask: "Type 1 in the chat if your check says READY, 2 if not." |
| 12:00 to 12:10 | Setup block. Ask people who typed 2 to stay on the fallback path below. Everyone else starts the warm-up cell. |
| 12:10 | Part 1 begins. |

**Rule: two minutes per person, then move to the browser option.** A person who cannot get a working local environment in the live session switches to Binder or Colab, takes part in the workshop from there, and fixes the local install afterwards by email. This keeps one troubleshooting case from costing the group time.

During exercises:

- Ask participants to write "stuck" in the chat. Answer in the chat first; only talk through a problem on audio if several people have the same one.
- Ask experienced participants (invite this in the welcome) to help others in the chat or the Slack workspace. Do not rely on this.
- Keep a list of unresolved problems and follow up by email after the session.

## Session 2 (Oct 27)

Same structure, with a shorter setup block (5 minutes), since the environment was proven in session 1. The most likely problems are people who use Colab or Binder (the session reset, so they must run the setup cell again) and people who missed session 1.

## After each session

- Send the recording and materials (see the follow-up email template, when written).
- Answer the unresolved-problem list within two days.
- Note recurring failures and add them to the common-errors table in `setup/INSTALL.md`.

## Decisions to confirm

- Whether the 10 minute setup block should come out of the 2 hours (it shortens each part to about 50 minutes) or the sessions should run slightly over. The plan assumes it comes out of the 2 hours.
- Whether Zoom early entry at 11:45 is acceptable for you.
