# ③ 草案メモ：日本語音声認識の前処理としてのRE-USE

状態：未着手。計画のメモ。

## 仮タイトル

- NVIDIAの音声強調モデルRE-USEは、日本語音声認識の精度を上げるのか試してみた

## 問い

- 劣化した日本語音声にRE-USEをかけると、ASRのCERは下がるか
- きれいな音声にかけたとき、CERが悪化しないか（学習データに日本語は含まれない）
- 劣化の種類（雑音、残響、帯域制限、コーデック）ごとに効果は違うか
- ASRモデルによって効果が違うか（Whisper系のように雑音に強いモデルでは効果が小さい可能性）

## 前回記事から流用できるもの

- 前回記事：[NVIDIA L40Sで日本語対応ASRモデルを比較してみた](https://dev.classmethod.jp/articles/japanese-asr-l40s-fleurs-benchmark/)
- リポジトリ：`../japanese-asr-l40s-benchmark`
  - FLEURS ja_jp test 650件の準備と検証（`prepare_fleurs.py`、`validate_manifest.py`）
  - 正規化とCER算出（`normalize_ja.py`、`calculate_cer.py`）
  - 7構成のASR実行環境とBrev上の手順
- 前回のクリーン音声でのCER（基準値）：Whisper large-v3 4.49%、large-v3-turbo 4.81%、Parakeet 5.94%、Kotoba v2.2 6.92%、ReazonSpeech 9.40%、Nemotron offline 13.31%

## 実験案

- 対象ASR：前回の上位から2〜3構成（Whisper large-v3、Parakeet、可能ならReazonSpeech）
- 入力条件
  - クリーン（FLEURSそのまま）
  - 加法性雑音：DNS Challengeなどの雑音をSNR 0、5、10、20dBで混ぜる
  - 残響：RIRを畳み込む
  - 帯域制限：8kHzにダウンサンプルしてから16kHzに戻す（電話音声を想定）
  - コーデック：低ビットレートのMP3やOpus
- 各条件で「RE-USEなし」と「RE-USEあり」を比べる
- RE-USEの実行環境：L40Sの公式実装（速い）。Mac版も使えるので、Macでの事前処理という形も選べる
- 余力があれば：自分で録音した雑音や残響のある日本語音声（定性評価）、笑い声保持版との比較

## 考えておくこと

- FLEURSは読み上げ音声で、もともとかなりきれい。前処理の効果を見るには劣化を人工的に加える必要がある
- 劣化を加える乱数の種と雑音源を固定し、条件を再現できるようにする
- RE-USEは帯域拡張や雑音除去で音を「作る」ので、ASRには逆効果の可能性もある（幻聴に近い挙動）。悪化した例も具体的に示す
- NemotronのモデルカードはFLEURSを学習データに含むと明記している（前回と同じ注意）
- 雑音データとRIRのライセンスを確認する
- 処理時間も記録して、前処理込みのスループットを示す

## TODO

- [ ] 劣化条件と雑音源を決める
- [ ] Brevのworkspaceを決める（`cm-nakamura-asr-benchmark` に前回のASR環境が残っている。`cm-nakamura-re-use` にはRE-USEの環境がある）
- [ ] 前回リポジトリのスクリプトに、前処理のステップを差し込む形で設計する
