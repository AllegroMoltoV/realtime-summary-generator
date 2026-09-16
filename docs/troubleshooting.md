# トラブルシューティング

## 確認する情報

問題が発生した場合は、次の情報を確認します。

- 発生時刻
- OBS とコンソールに現れた症状
- `.logs/realtime-summary.jsonl` の同時刻付近
- 使用した入力デバイス
- OBS Studio と Python のバージョン

診断ログは発話本文と認証情報を記録しません。ただし、音声デバイス名とスタックトレース内のローカルファイルパスを含む場合があります。公開された GitHub Issue へ添付する前に内容を確認してください。

## 詳細ログ

通常のログレベルは `INFO` です。問題を再現できる場合は、`.env` の任意設定を有効にしてから起動します。

```dotenv
LOG_LEVEL=DEBUG
```

直近 50 件は PowerShell で確認できます。

```powershell
Get-Content .logs\realtime-summary.jsonl -Tail 50
```

確認後は `LOG_LEVEL=DEBUG` を削除するか、行頭へ `#` を付けて無効にできます。

## 起動直後に終了する

次の順序で確認します。

1. リポジトリのルートで起動したか確認します。別のディレクトリから起動すると `.env` を読み込めません。
2. `.env` に `OPENAI_API_KEY`、`GEMINI_API_KEY`、`OBS_WEBSOCKET_PASSWORD` があるか確認します。
3. OBS が起動しているか確認します。
4. OBS の WebSocket サーバーが有効で、ポートが `4455` か確認します。
5. `.env` の OBS WebSocket パスワードが OBS の設定と一致しているか確認します。
6. 三つの Text (GDI+) ソース名が完全に一致し、`Read from file` が無効か確認します。

必要なソース名は次のとおりです。

- `字幕（日本語）`
- `字幕（英語）`
- `概要（日本語）`

## 日本語字幕が表示されない

利用可能な入力デバイスを表示します。

```powershell
.\.venv\Scripts\python.exe -m sounddevice
```

意図したデバイスが Windows の既定入力でなければ、一覧の数値 ID を `.env` に設定します。

```dotenv
AUDIO_INPUT_DEVICE=12
```

デバイス ID は、再起動や音声機器の接続変更で変わる場合があります。設定後も表示されない場合は、Windows のプライバシー設定でデスクトップアプリのマイク利用が許可されていることと、音声インターフェースへ入力が届いていることを確認します。

ログでは、`microphone_started`、OpenAI の `connected`、`input_status`、`audio_chunks_dropped` を確認します。

## OpenAI との接続が切れる

一時的なネットワーク障害や WebSocket の切断が発生すると、最大 5 回まで自動で再接続します。待機時間は再試行ごとに長くなり、16 秒を上限とします。接続が 60 秒以上継続した場合は、連続再試行回数を 0 に戻します。

次のイベントと項目を時刻順に確認します。

- `connection_lost`: 切断または一時的な接続失敗を検出した記録
- `received_close_code`、`received_close_reason`: サーバーから受信した Close フレーム
- `sent_close_code`、`sent_close_reason`: クライアントから送信した Close フレーム
- `duration_ms`: 接続開始から切断検出までの時間
- `reconnect_scheduled`: 次の再接続を予約した記録
- `retry_count`、`retry_delay_seconds`: 再試行回数と待機時間
- `audio_chunks_discarded`: 再接続前に破棄した古い音声チャンク数
- `transcription_items_discarded`、`partial_captions_discarded`: 再接続前に破棄した未完了の発話数と認識途中の字幕数
- `timing_items_discarded`: 再接続前に破棄した処理時間計測中の発話数
- `reconnect_exhausted`: 最大再試行回数へ到達した記録
- `ping_interval_seconds`、`ping_timeout_seconds`: 接続時に指定した WebSocket の Ping 間隔とタイムアウト
- `max_retries`、`retry_base_delay_seconds`、`retry_max_delay_seconds`、`stable_connection_seconds`: 実行時の再接続ポリシー

Close フレームを受信しないまま接続が失われた場合は、`received_close_code` が `null`、`error_code` が `1006` になります。再接続中に蓄積した音声は破棄するため、切断中の発話は字幕へ反映されません。

認証エラーや API が通知した要求エラーは、同じ要求を繰り返しても回復しないため再試行しません。コンソールに表示されたエラーと同時刻のログを確認してください。

## 英語字幕が表示されない

日本語字幕が表示される場合は、音声入力と OpenAI の接続は動作しています。次を確認します。

1. `GEMINI_API_KEY` が有効か確認します。
2. Gemini API を利用できるプロジェクトと請求設定か確認します。
3. ログで `translation_failed` を検索します。

英訳は確定した発話ごとに生成されます。話し続けている間や無音区間がない場合は、発話の確定まで更新されないことがあります。

## 日本語概要が表示されない

概要は、新しい確定発話がある場合に約 20 秒間隔で更新します。起動直後、無音中、新しい発話がない間は更新されません。

英語字幕も表示されない場合は Gemini API の設定を確認します。英語字幕だけ表示される場合は、ログで `summary_failed`、`buffer_cycles`、`dropped_cycles` を確認します。

## OBS の表示だけ更新されない

OBS の各ソースが Text (GDI+) で、`Read from file` が無効か確認します。ソースを作り直した場合は、名前が完全に一致しているか確認します。

ログで `preflight_completed` があれば、起動時のソース確認は成功しています。`text_update_failed` があれば、その時点で OBS 更新に失敗しています。

## 表示が遅れる

日本語字幕は OpenAI が返す認識差分、英語字幕は確定発話の翻訳、概要は約 20 秒ごとの生成結果です。三つの表示は同時には更新されません。

ログの次の項目から処理の詰まりを確認できます。

- `audio_chunks_dropped`: 音声送信が入力へ追い付かなかったチャンク数
- `translation_completed.duration_ms`: 英訳の処理時間
- `summary_completed.duration_ms`: 概要生成の処理時間
- `buffer_cycles`: 概要生成を待つ周期数
- `dropped_cycles`: 保持上限により破棄した概要周期数

## 終了後も文字が残る

通常終了時は三つのソースを空にします。プロセスを強制終了した場合や OBS との接続が切れた場合は、文字が残ることがあります。OBS WebSocket を有効にしてアプリを再起動すると、開始時に三つの表示を空にします。

## 更新情報

- 0.1.0 (2026-09-13): 初版を公開。
