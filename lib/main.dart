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

// uri
const String kVoiceChatEndpoint = "https://brunswick-bloomberg-floppy-san.trycloudflare.com/voice-chat";

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

class VoiceChatPage extends StatefulWidget {
  const VoiceChatPage({super.key, required this.title});
  final String title;

  @override
  State<VoiceChatPage> createState() => _VoiceChatPageState();
}

class _Message {
  final String text;
  final bool isUser;
  _Message(this.text, this.isUser);
}

class _VoiceChatPageState extends State<VoiceChatPage> {
  final FlutterSoundRecorder _recorder = FlutterSoundRecorder();
  bool _isRecording = false;
  String? _recordedPath;

  final List<_Message> _messages = [];
  late String _sessionId;

  @override
  void initState() {
    super.initState();
    _initRecorder();
    _newChat();
  }

  String _makeSessionId() {
    final now = DateTime.now().microsecondsSinceEpoch;
    final rnd = Random().nextInt(1 << 32);
    return 'sess-$now-$rnd';
  }

  void _newChat() {
    setState(() {
      _sessionId = _makeSessionId();
      _messages.clear();
    });
  }

  Future<void> _initRecorder() async {
    await _requestPermission();
    await _recorder.openRecorder();
  }

  Future<void> _requestPermission() async {
    final micStatus = await Permission.microphone.request();
    if (!micStatus.isGranted) {
      throw RecordingPermissionException('🎙️ Microphone permission not granted');
    }

    if (Platform.isAndroid) {
      final notif = await Permission.notification.request();
      if (!notif.isGranted) {
        debugPrint('⚠️ Notification permission not granted.');
      }
    }
  }

  DateTime? _recordStartTime;

  Future<void> _startRecording() async {
    final dir = await getTemporaryDirectory();
    _recordedPath = '${dir.path}/recorded.aac';

    if (!_recorder.isRecording) {
      if (!_recorder.isPaused && !_recorder.isStopped) {
        await _recorder.openRecorder();
      }
    }

    await _recorder.startRecorder(
      toFile: _recordedPath!,
      codec: Codec.aacADTS,
      sampleRate: 44100,
      numChannels: 1,
      audioSource: AudioSource.microphone,
    );

    _recordStartTime = DateTime.now();
    setState(() {
      _isRecording = true;
    });
  }

  Future<void> _stopRecordingAndSend() async {
    await _recorder.stopRecorder();
    await Future.delayed(const Duration(milliseconds: 100));
    setState(() => _isRecording = false);

    if (_recordedPath == null || !File(_recordedPath!).existsSync()) {
      setState(() => _messages.add(_Message("❌ 녹음된 파일이 존재하지 않습니다.", false)));
      return;
    }

    try {
      final cleanUrl = kVoiceChatEndpoint
          .trim()
          .replaceAll(RegExp(r'[\u200B-\u200D\u2060\uFEFF]'), '');
      final uri = Uri.parse(cleanUrl);

      final req = http.MultipartRequest('POST', uri)
        ..headers['X-Session-Id'] = _sessionId
        ..files.add(await http.MultipartFile.fromPath('file', _recordedPath!));

      final streamed = await req.send();
      final body = await streamed.stream.bytesToString(); // default: utf8

      if (streamed.statusCode == 202) {
        final m = jsonDecode(body) as Map<String, dynamic>;
        final jobId = (m['job_id'] ?? '').toString();
        if (jobId.isEmpty) {
          setState(() => _messages.add(_Message("❌ job_id가 비어 있습니다.", false)));
          return;
        }
        setState(() => _messages.add(_Message("자료 검색중...", false)));
        await _pollResult(jobId);
        return;
      }

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

  // --- [FIX] UTF-8 강제 디코딩으로 문자깨짐 해결 ---
  Future<void> _pollResult(String jobId) async {
    final base = Uri.parse(
      kVoiceChatEndpoint.trim().replaceAll(RegExp(r'[\u200B-\u200D\u2060\uFEFF]'), ''),
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
        // ✅ 핵심: 응답을 바이트로 받아 UTF-8로 직접 디코딩
        final raw = utf8.decode(r.bodyBytes);
        final m = jsonDecode(raw) as Map<String, dynamic>;
        final status = (m['status'] ?? '').toString();

        if (status == 'done') {
          final question = (m['question'] ?? '').toString();
          final answer   = (m['answer'] ?? '').toString();
          setState(() {
            if (question.isNotEmpty) _messages.add(_Message(question, true));
            _messages.add(_Message(answer.isNotEmpty ? answer : "❔ 응답이 비어 있습니다.", false));
          });
          return;
        }
        if (status == 'error') {
          final msg = (m['message'] ?? '처리 중 오류가 발생했습니다.').toString();
          setState(() => _messages.add(_Message("❌ $msg", false)));
          return;
        }
        // status == 'processing' 이면 계속 대기
      } else {
        // 에러 바디도 UTF-8로 강제 디코딩
        final err = utf8.decode(r.bodyBytes);
        setState(() => _messages.add(_Message("❌ 결과 조회 실패: ${r.statusCode} $err", false)));
        return;
      }

      if (DateTime.now().difference(start) > maxWait) {
        setState(() => _messages.add(_Message("❌ 처리 지연: 잠시 후 다시 시도해 주세요.", false)));
        return;
      }
      await Future.delayed(const Duration(seconds: 1));
    }
  }
  // --- [FIX] 끝 ---

  @override
  void dispose() {
    _recorder.closeRecorder();
    super.dispose();
  }

  Widget _bubble(_Message m) {
    final isUser = m.isUser;
    final bubbleColor = isUser ? Colors.indigo.shade300 : Colors.grey.shade200;
    final textColor = isUser ? Colors.white : Colors.black87;

    return Align(
      alignment: isUser ? Alignment.centerRight : Alignment.centerLeft,
      child: ConstrainedBox(
        constraints: BoxConstraints(
          maxWidth: MediaQuery.of(context).size.width * 0.75,
        ),
        child: Container(
          padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
          margin: const EdgeInsets.symmetric(vertical: 4),
          decoration: BoxDecoration(
            color: bubbleColor,
            borderRadius: BorderRadius.only(
              topLeft: const Radius.circular(14),
              topRight: const Radius.circular(14),
              bottomLeft: Radius.circular(isUser ? 14 : 4),
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

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(widget.title),
        actions: [
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
            Row(
              children: [
                Expanded(
                  child: ElevatedButton.icon(
                    onPressed: _isRecording ? null : _startRecording,
                    icon: const Icon(Icons.mic),
                    label: const Text('녹음 시작'),
                  ),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: ElevatedButton.icon(
                    onPressed: _isRecording ? _stopRecordingAndSend : null,
                    icon: const Icon(Icons.stop_circle_outlined),
                    label: const Text('녹음 종료 및 전송'),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 12),
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
