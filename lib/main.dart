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
  String _makeSessionId(){
    final now = DateTime.now().microsecondsSinceEpoch;
    final rnd = Random().nextInt(1 << 32);
    return 'sess-$now-$rnd';
  }
  
  void _newChat(){
    setState((){
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
    // Android 13 미만에서도 호출해도 안전: plugin이 내부에서 처리
      final notif = await Permission.notification.request();
      if (!notif.isGranted) {
        // 포그라운드 서비스 알림 못 띄우면 녹음이 즉시 종료될 수 있음
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

    // await _recorder.startRecorder(
    //   toFile: _recordedPath,
    //   codec: Codec.pcm16WAV,          // WAV 원하면 유지
    //   sampleRate: 44100,              // 기기 호환 좋은 값 44100
    //   numChannels: 1,                 // 모노
    //   audioSource: AudioSource.microphone, // 명시적으로 마이크
    // );
    await _recorder.startRecorder(
      toFile: _recordedPath!.replaceAll('.wav', '.aac'),
      codec: Codec.aacADTS,   // ← 임시로 AAC
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
    final duration = DateTime.now().difference(_recordStartTime ?? DateTime.now());
    print('⏱ 녹음 시간: ${duration.inMilliseconds} ms');
    if (_recordedPath == null || !File(_recordedPath!).existsSync()) {
      setState(() => _messages.add(_Message("❌ 녹음된 파일이 존재하지 않습니다.", false)));
      return;
    }

    final file = File(_recordedPath!);
    final fileSize = await file.length();
    print('📂 저장된 파일: $_recordedPath (크기: $fileSize bytes)');

    try {
      var request = http.MultipartRequest(
        'POST',
        // 실제 물리 디바이스라면 → Uri.parse("http://<your_local_ip>:8000/voice-chat")
        Uri.parse("http://192.168.0.4:8000/voice-chat"), // 실제 서버 주소
        //Uri.parse("http://10.0.2.2:8000/voice-chat"), // Android emulator에서는 10.0.2.2 사용
      );

      request.headers['X-Session-Id'] = _sessionId;

      request.files.add(await http.MultipartFile.fromPath('file', _recordedPath!));

      var response = await request.send();

      if (response.statusCode == 200) {
        final respStr = await response.stream.bytesToString();
        final jsonResp = jsonDecode(respStr);
        final question = (jsonResp['question'] ?? '').toString();
        final answer = (jsonResp['answer'] ?? '').toString();
        
        setState(() {
          if (question.isNotEmpty){
            _messages.add(_Message(question, true));
          }
          _messages.add(_Message(answer.isNotEmpty ? answer : "❔ 응답이 비어 있습니다.", false));
        });
      } else {
        setState(() => _messages.add(_Message("❌ 서버 오류: ${response.statusCode}", false)));
      }
    } catch (e) {
      setState(() => _messages.add(_Message("❌ 전송 실패: $e", false)));
    }
  }

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
                color: Colors.black.withValues(alpha: 0.05),
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
            onPressed: _newChat, // 히스토리 비우고 새 session_id 생성
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
