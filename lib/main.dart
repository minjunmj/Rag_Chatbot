import 'package:flutter/material.dart';
import 'package:flutter_sound/flutter_sound.dart';
import 'package:path_provider/path_provider.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:http/http.dart' as http;
import 'dart:io';
import 'dart:convert';
import 'dart:math';

void main() {
  runApp(const MyApp());
}

// 서버 엔드포인트 (Cloudflare Tunnel이 생성하는 공개 URL)
// 서버를 재시작하면 URL이 바뀌므로, 변경 시 이 값을 업데이트해야 함
const String kVoiceChatEndpoint =
    "https://brunswick-bloomberg-floppy-san.trycloudflare.com/voice-chat";

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Voice Chatbot',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.deepPurple),
        useMaterial3: true,
      ),
      home: const VoiceChatPage(title: 'Voice Chatbot Home'),
    );
  }
}

// ─────────────────────────────────────────────
// 화면 위젯
// ─────────────────────────────────────────────
class VoiceChatPage extends StatefulWidget {
  const VoiceChatPage({super.key, required this.title});
  final String title;

  @override
  State<VoiceChatPage> createState() => _VoiceChatPageState();
}

// 채팅 메시지 모델 (텍스트 + 발신자 구분)
class _Message {
  final String text;
  final bool isUser; // true: 사용자(오른쪽 버블), false: AI(왼쪽 버블)
  _Message(this.text, this.isUser);
}

class _VoiceChatPageState extends State<VoiceChatPage> {
  // flutter_sound 녹음기 인스턴스
  final FlutterSoundRecorder _recorder = FlutterSoundRecorder();
  bool _isRecording = false; // 현재 녹음 중 여부
  String? _recordedPath;    // 녹음 파일 임시 저장 경로

  final List<_Message> _messages = []; // 화면에 표시할 메시지 목록
  late String _sessionId;              // 서버와 대화 이력을 공유하는 세션 ID

  @override
  void initState() {
    super.initState();
    _initRecorder(); // 녹음기 초기화 및 권한 요청
    _newChat();      // 새 세션 ID 생성 + 메시지 초기화
  }

  // 타임스탬프 + 난수로 고유 세션 ID 생성
  String _makeSessionId() {
    final now = DateTime.now().microsecondsSinceEpoch;
    final rnd = Random().nextInt(1 << 32);
    return 'sess-$now-$rnd';
  }

  // 새 대화 시작: 세션 ID 갱신 + 메시지 목록 초기화
  void _newChat() {
    setState(() {
      _sessionId = _makeSessionId();
      _messages.clear();
    });
  }

  // 녹음기 열기 + 마이크/알림 권한 요청
  Future<void> _initRecorder() async {
    await _requestPermission();
    await _recorder.openRecorder();
  }

  // 플랫폼별 필수 권한 요청
  Future<void> _requestPermission() async {
    final micStatus = await Permission.microphone.request();
    if (!micStatus.isGranted) {
      throw RecordingPermissionException('🎙️ Microphone permission not granted');
    }

    // Android는 포그라운드 서비스 알림 권한도 필요
    if (Platform.isAndroid) {
      final notif = await Permission.notification.request();
      if (!notif.isGranted) {
        debugPrint('⚠️ Notification permission not granted.');
      }
    }
  }

  DateTime? _recordStartTime; // 녹음 시작 시각 (필요 시 최소 길이 체크용)

  // 녹음 시작: AAC 형식으로 임시 디렉토리에 저장
  Future<void> _startRecording() async {
    final dir = await getTemporaryDirectory();
    _recordedPath = '${dir.path}/recorded.aac';

    // 이미 열려있지 않으면 recorder 재오픈
    if (!_recorder.isRecording) {
      if (!_recorder.isPaused && !_recorder.isStopped) {
        await _recorder.openRecorder();
      }
    }

    await _recorder.startRecorder(
      toFile: _recordedPath!,
      codec: Codec.aacADTS,    // AAC 포맷 (서버에서 ffmpeg로 WAV 변환)
      sampleRate: 44100,
      numChannels: 1,          // 모노
      audioSource: AudioSource.microphone,
    );

    _recordStartTime = DateTime.now();
    setState(() {
      _isRecording = true;
    });
  }

  // 녹음 종료 후 서버로 파일 전송
  Future<void> _stopRecordingAndSend() async {
    await _recorder.stopRecorder();
    await Future.delayed(const Duration(milliseconds: 100)); // 파일 flush 대기
    setState(() => _isRecording = false);

    // 녹음 파일이 없으면 오류 메시지 표시
    if (_recordedPath == null || !File(_recordedPath!).existsSync()) {
      setState(() => _messages.add(_Message("❌ 녹음된 파일이 존재하지 않습니다.", false)));
      return;
    }

    try:
    // 보이지 않는 유니코드 문자(제로폭 공백 등) 제거 후 URI 파싱
      final cleanUrl = kVoiceChatEndpoint
          .trim()
          .replaceAll(RegExp(r'[​-‍⁠﻿]'), '');
      final uri = Uri.parse(cleanUrl);

      // multipart/form-data로 음성 파일 전송
      // X-Session-Id 헤더로 서버의 대화 이력과 연결
      final req = http.MultipartRequest('POST', uri)
        ..headers['X-Session-Id'] = _sessionId
        ..files.add(await http.MultipartFile.fromPath('file', _recordedPath!));

      final streamed = await req.send();
      final body = await streamed.stream.bytesToString();

      // 202: 비동기 처리 시작 → job_id로 폴링
      if (streamed.statusCode == 202) {
        final m = jsonDecode(body) as Map<String, dynamic>;
        final jobId = (m['job_id'] ?? '').toString();
        if (jobId.isEmpty) {
          setState(() => _messages.add(_Message("❌ job_id가 비어 있습니다.", false)));
          return;
        }
        // 폴링 중 임시 "검색중" 메시지 표시
        setState(() => _messages.add(_Message("자료 검색중...", false)));
        await _pollResult(jobId);
        return;
      }

      // 200: 동기 처리 결과 (현재 서버는 항상 202 반환)
      if (streamed.statusCode == 200) {
        final obj = jsonDecode(body) as Map<String, dynamic>;
        final question = (obj['question'] ?? '').toString();
        final answer   = (obj['answer'] ?? '').toString();

        setState(() {
          if (question.isNotEmpty) _messages.add(_Message(question, true));
          _messages.add(_Message(
              answer.isNotEmpty ? answer : "❔ 응답이 비어 있습니다.", false));
        });
        return;
      }

      setState(() => _messages.add(_Message("❌ 서버 오류: ${streamed.statusCode} $body", false)));
    } catch (e) {
      setState(() => _messages.add(_Message("❌ 전송 실패: $e", false)));
    }
  }

  // ─────────────────────────────────────────────
  // 비동기 작업 결과 폴링 (1초 간격, 최대 3분 대기)
  // 한글 깨짐 방지: r.bodyBytes를 utf8.decode로 직접 디코딩
  // ─────────────────────────────────────────────
  Future<void> _pollResult(String jobId) async {
    // /voice-chat → /voice-chat/result 로 경로 변환
    final base = Uri.parse(
      kVoiceChatEndpoint.trim().replaceAll(RegExp(r'[​-‍⁠﻿]'), ''),
    );

    final resultUri = base.replace(
      path: base.path.replaceFirst(RegExp(r'/voice-chat/?$'), '/voice-chat/result'),
      queryParameters: {'job_id': jobId},
    );

    const maxWait = Duration(minutes: 3);
    final start = DateTime.now();

    while (true) {
      final r = await http.get(resultUri, headers: {'X-Session-Id': _sessionId});

      if (r.statusCode == 200) {
        // bodyBytes → utf8 강제 디코딩 (한글 깨짐 방지)
        final raw = utf8.decode(r.bodyBytes);
        final m = jsonDecode(raw) as Map<String, dynamic>;
        final status = (m['status'] ?? '').toString();

        if (status == 'done') {
          // 처리 완료: 질문 + 답변을 채팅 버블에 추가
          final question = (m['question'] ?? '').toString();
          final answer   = (m['answer'] ?? '').toString();
          setState(() {
            if (question.isNotEmpty) _messages.add(_Message(question, true));
            _messages.add(_Message(answer.isNotEmpty ? answer : "❔ 응답이 비어 있습니다.", false));
          });
          return;
        }

        if (status == 'error') {
          // 서버 오류 메시지 표시
          final msg = (m['message'] ?? '처리 중 오류가 발생했습니다.').toString();
          setState(() => _messages.add(_Message("❌ $msg", false)));
          return;
        }
        // status == 'processing': 계속 대기
      } else {
        final err = utf8.decode(r.bodyBytes);
        setState(() => _messages.add(_Message("❌ 결과 조회 실패: ${r.statusCode} $err", false)));
        return;
      }

      // 최대 대기 시간 초과
      if (DateTime.now().difference(start) > maxWait) {
        setState(() => _messages.add(_Message("❌ 처리 지연: 잠시 후 다시 시도해 주세요.", false)));
        return;
      }

      await Future.delayed(const Duration(seconds: 1)); // 1초 대기 후 재시도
    }
  }

  @override
  void dispose() {
    _recorder.closeRecorder(); // 화면 종료 시 녹음기 자원 해제
    super.dispose();
  }

  // ─────────────────────────────────────────────
  // 채팅 버블 위젯 빌더
  // ─────────────────────────────────────────────
  Widget _bubble(_Message m) {
    final isUser = m.isUser;
    // 사용자: 남색 배경 / AI: 밝은 회색 배경
    final bubbleColor = isUser ? Colors.indigo.shade300 : Colors.grey.shade200;
    final textColor   = isUser ? Colors.white : Colors.black87;

    return Align(
      alignment: isUser ? Alignment.centerRight : Alignment.centerLeft,
      child: ConstrainedBox(
        constraints: BoxConstraints(
          maxWidth: MediaQuery.of(context).size.width * 0.75, // 최대 화면 폭의 75%
        ),
        child: Container(
          padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
          margin: const EdgeInsets.symmetric(vertical: 4),
          decoration: BoxDecoration(
            color: bubbleColor,
            borderRadius: BorderRadius.only(
              topLeft:     const Radius.circular(14),
              topRight:    const Radius.circular(14),
              // 발신자 쪽 하단 모서리를 뾰족하게 (말풍선 효과)
              bottomLeft:  Radius.circular(isUser ? 14 : 4),
              bottomRight: Radius.circular(isUser ? 4 : 14),
            ),
            boxShadow: [
              BoxShadow(
                blurRadius: 2,
                spreadRadius: 0,
                offset: const Offset(0, 1),
                color: Colors.black.withOpacity(0.05),
              ),
            ],
          ),
          child: Text(m.text, style: TextStyle(color: textColor, height: 1.3)),
        ),
      ),
    );
  }

  // ─────────────────────────────────────────────
  // 메인 UI 빌드
  // ─────────────────────────────────────────────
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(widget.title),
        actions: [
          // 새 대화 버튼: 세션 초기화 + 메시지 목록 비우기
          IconButton(
            tooltip: '새 대화 시작',
            icon: const Icon(Icons.refresh),
            onPressed: _newChat,
          ),
        ],
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          children: [
            // 녹음 제어 버튼 (녹음 시작 / 종료 및 전송)
            Row(
              children: [
                Expanded(
                  child: ElevatedButton.icon(
                    // 녹음 중이면 비활성화
                    onPressed: _isRecording ? null : _startRecording,
                    icon: const Icon(Icons.mic),
                    label: const Text('녹음 시작'),
                  ),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: ElevatedButton.icon(
                    // 녹음 중일 때만 활성화
                    onPressed: _isRecording ? _stopRecordingAndSend : null,
                    icon: const Icon(Icons.stop_circle_outlined),
                    label: const Text('녹음 종료 및 전송'),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 12),
            // 채팅 메시지 목록 (스크롤 가능)
            Expanded(
              child: ListView.builder(
                padding: const EdgeInsets.only(top: 4),
                itemCount: _messages.length,
                itemBuilder: (context, index) => _bubble(_messages[index]),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
