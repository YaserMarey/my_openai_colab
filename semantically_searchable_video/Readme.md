### Semantically Searchable Arabic Language Youtube Video

Whisper is an automatic speech recognition (ASR) system developed by
OpenAI. It has been trained on 680,000 hours of multilingual and
multitasking supervised data collected from the web. This large and
diverse dataset leads to improved robustness to accents, background
noise, and technical language.

But how does it perform in Arabic?

according to [whisper](https://github.com/openai/whisper) \"Whisper\'s
performance varies widely depending on the language.\" with WER (Word
Error Rate) of 15.7 for Arabic compared to 4.2 for English.

WER is calculated by comparing the transcribed text produced by the
speech recognition system to a reference transcription and counting the
number of substitutions, deletions, and insertions required to align the
two texts. The WER is then calculated as the total number of errors
divided by the total number of words in the reference transcription.

In this notebook, I will try to ad-hock test Whisper\'s performance in
Arabic.

I will apply Whisper to transcript a [short
video](https://www.youtube.com/watch?v=RgG5e87-Lbs). The
[video](https://www.youtube.com/watch?v=RgG5e87-Lbs) will be in Arabic
with modern Egyptian Dialect which usually contains English words or
sentences. I will then embed text segments and overlap them and then
query these segments by calculating the cosine similarity.

Found segments will then point us to the place in the video relevant to
the query.

#### Steps:

-   Download the video from YouTube
-   Transcript video using Whisper
-   Combine segments for more meaningful chunks
-   Text embed transcripted segments
-   Text embed query
-   Calculate similarity

##### First let\'s install dependencies

``` python
!pip install openai pytube tiktoken
```

``` python
!pip install git+https://github.com/openai/whisper.git 
```

##### Now download the video and extract the audio track. {#now-download-the-video-and-extract-the-audio-track}

``` python
import whisper
from pytube import YouTube
youtube_video_url = "https://www.youtube.com/watch?v=RgG5e87-Lbs"
youtube_video = YouTube(youtube_video_url)
audio = youtube_video.streams.get_audio_only().download()
```

#### load the smallest model ( base ) and transcript the extracted audio in the step above. {#load-the-smallest-model--base--and-transcript-the-extracted-audio-in-the-step-above}

``` python
model = whisper.load_model('base')
transcript_output = model.transcribe(audio)
transcript_output
```
    100%|████████████████████████████████████████| 139M/139M [00:01<00:00, 104MiB/s]

    {'text': ' الزيتم تأسيري الاسبلط سكرين على دوبر من يرقرون يلّبين هذا أول حاجة نعملها ينجيب بيدي هنا لحنا نسردهم من حطهم فوقات ويزعمنا عاديك وبعدك ده نختار الفيديو اللي فوق نروح على فكت كنترولز ونلعب في البوزيشة اللغات مقابق في نصششة بالزلط حرى كده بالسة ونجيبوا لغائطه فبكنا كنت عسنة الشاشة نفسي بس في مشكلة إنك في جوزء من الفيديو نحت اللي منها نحفنا عايزين يصور عايزين نظر مسلم حيث تلعلي مندي في الحال مشكلة ده أنا جاهت أن يزعم كامت وبعدك ده نروح على فكت سيكون هناك ده نعمل السشة على فكت اسمه رب نحو ده الافكت ونعملوا دراجن رب والخطو على الفيديو اللي فوق ونقذو الحالت دي الفيديو اللي فوق نظر طالي ونروح على فكت قنترولز حلق إيمان جود هنئي في فكت قنترولز بالشكل ذكه اللي عندنا عايز ارباء فشنستحط بعض كده يوم اللي فتوطب رايد تبطه كروب يحن تترطى حتى من الفيديو فنته بيولى بعيد تطعم انه حتى من الشمال واللمين واللمين فوق اللي من بحطة فانقام الشمال شوية اللي قطم حسنوية يضظمقت يعني بالشكل ذكه وبعدك ده نطالفو نروح على البزيشن نوم رجعينه تنئي نصم طروح ونص البديو وبعدك ده بعد البعي ساح لمنكن نروحنا من سلكت الفيديو التنئي ونروحة البزيشن نحر راكو الغاية من نظهر جوز اللي حيث نحن نقنين البديو بنفس الشكي حلت بل المتحن بسنعنيني بشكل ذكه نقرب مشاكل الفيديو جدا اللي يكل الفيديو شغيل في جوز اتعوم الشيشن وكذن كنش رحنا الزنام الاتسللس بلسكري طبعا بل أن تأتي استعمل حقظة يفاصل أو خطئ يفسر من الملبيديو هيين ممكن نفسهم خطهم خطهم ملبيديو هيين عشاني بشكل وحسن كنش رحنا الطريق بتفسيل فيديو ابل كدهات لول لنك في الأيفو فروحة البنتول هل هيها سدس تول متأحن نضع طاليها وبعدك ده نروح الأول فيديو فوحة اللي حيث لحايا زين نسمى النضل خطت نضع ضغطتك لك نوز نرسم في اخر الفيديو تحت اي كبن نقطة اللي اي رثنى الخططة عنها بالشكل بحي طاعة انبات لايش شكل كونوات الى الفو البيديو رحنا طارة السليكشن طول نمس سليكت الفيديو نفضطها على ادل الفيديو بنين بزنطة الشكل بحي طاعة انبان تمكت العب في شكل خططة من الفيكت كنتورلس تعمل سليكت الشكل وطروح الفيكت كنتورلس تليقي في شيب هناك كون احوة اتلاقي باك كل حاجة لعلى قبيعة لأي في فلو الوستروك ونمكن بضغير اللون زي من تعيز ونمكن نضطين كده ونختر اللون نننن تعيز ونرسل مثل عبيض ونمكن يعمل تضطة الوستروك لستروك ده البردر بتاعة الشكل لك انبردر عبيضه ونمكن يزوود حاجم عشان اي كبن الخططة بالشكل لك شان يزهر بس اكت ونقرب عشان الفيديو اللي اي بعب الشكل ونقرب كل حاجة لنا نحيت الحلامة سوشان في الفيديو تشركوا في القناة ونقصمنا في الفيس بوك ونقصمنا في الفيس بوك سلام',
     
The quality of the transcript is not good at all, no need to even count
WER.


Now let\'s try the rest of the models, Small, Medium, and Large.

``` python
model = whisper.load_model('small')
transcript_output = model.transcribe(audio)
transcript_output
```

    100%|███████████████████████████████████████| 461M/461M [00:06<00:00, 78.0MiB/s]

    {'text': ' إذا تفعل تأسير الهاتف لإشتراك على أدوة برميير برو يلو بنا أول حاجة نجيب فيديو نستخدمهم ونضعهم فيه وكما نفعل وبعد ذلك نختار الفيديو ونذهب إلى افكت كونترولز ونلعب في الوضع اللغات ما يبقى في نص الشاشة بالضبط حراك هذا بسهل نجيبه اللغاية بها يمكن أن نكون عسنة الشاشة نصي بس في مشكلة إنك في جزء من الفيديو نحيت الليمين ها لأنها خفنة عايزين ويظهر عايزين نظر مثل حتى الليمين دي فأي الحل المشكلة دي نرجحة نزي مكانك وبعد ذلك نذهب إلى افكت اللي هي هنا كذا نمسش على افكت اسمه روب ناخد الافكت ونعملوا دراجن روب ونخطوه على الفيديو اللي فوق ونركزوا الحلتة دي ونفوق نضط عليه ونروح الافكت كونترولز حلقي موجود هنا ايه الافكت كونترولز بشكل ذاك حلقي عندنا ايه اربع افشان سحت باعة كذا اليوم اللفت وطب ورايت وبوتو كروبي يعني انت ترطع حتى من الفيديو فانتر بيقولك بعاية ترطع من انو حتى من الشمال واللميمين والمن فوق والمن تحت فاناقع من الشمال شوية اللي اللي نحس من هويه اتضمط يعني بشكل ذاك وبعد كذا نطلع فوق نروح الافكت ونمرجعينه تانيه نسانطروه ونص الفيديو وبعد كذا بقى البعيسة لممكن نروح نعمل سلقت الفيديو التاني وروح الافكت نحركه لغات من نظر الجزء اللي احنا عايزينه من الفيديو بنفس الشكل مثلا انا ايه كذا بالشكل دا نجرب نشغل الفيديو كذا حلقي كلي فيديو شغيل في الجزء اتعوم من الشاشة وكذا نكون شرحنا الزياني عملت اصليت في الاسكري طبعا لو انت عايز تعمل حاجة زي فاصل او بخط يصل من الفيديو هين ممكن نفسهم خط نحطوا من الفيديو عشاني بشكل أحسن وكننا شرحنا طريقة بتفصيلها فيديو قبل كذا اتلقوا اللينك في الافو فاروح على البينطول هل ايه سدس كل متعب انا اضط عليها وبعد كذا نروح الاول فيديو فوق بحيث اللي احنا عايزين نضع الضخط نضع الضخط نقل الناس ونرسم في اخر الفيديو تحت ايه كامل نقطة هل ايه راته من الخط التعنى بالشكل ده طبعا بات لعيش شكل كونه هاتي ايه فوق الفيديو اروح نختار السليكشن تول انا ما سليكت الفيديو نحطها على ادل فيديو بالشكل ده طبعا بان تمكن تلعب في شكل الخط ده من الفيكت كونترول ستعمل سليكت للشك وتروح الانفكت كونترولز هتلاقي في شيب هناك كون احو هتلاقي بك كل حاجة العلاق بي هلاقي في فلو وستروك ممكن بات غير اللون زي ما انت عايز ممكن نضع تناك ده ونختار اللون انت عايز وصراسي مثل ابيض ممكن بياما انتضع للستروك للستروك ده البردر بتاع الشكل من هالكا بردر ابيض وممكن نزود حاج معشان اين كبر الخطط بالشكل ده شاكا اني بس اقت وبس كده نجربه ان شغل الفيديو هلاقي بقب الشكل وكده كنو صنى ان هات الحلال ما شوش عمي لايك الفيديو وششركو في القناة اتبنا عصف حت الفيسبوك وبس كده سلام',
Better, but still not acceptable.

``` python
model = whisper.load_model('medium')
transcript_output = model.transcribe(audio)
transcript_output
```

    100%|█████████████████████████████████████| 1.42G/1.42G [00:18<00:00, 84.7MiB/s]

    {'text': ' كيف تقوم بتأثير الـ Split Screen على Adobe Premiere Pro لنبدأ أولا سنقوم بإستخدام فيديو هنا ونضعه على بعض المقاطع كما قمنا بفعل ثم نختار الفيديو الأعلى ونذهب إلى Effect Controls ونلعب في مكان الغاية المتصل في نصف الشاشة نحرك هذا بالسهل ونضعه على الغاية هنا يمكننا أن نكون أسنانا الشاشة نصف ولكن هناك مشكلة هناك جزء من الفيديو الأعلى ونحن نريد أن يصوره نريد أن نصور مثلاً هذه الأعلى فماذا نقوم بإجراء هذه المشكلة؟ نعود إلى مكان الغاية المتصل ثم نذهب إلى Effects نضعه هنا نقوم بإستخدام Effect name Crop نأخذ Effect ونضعه على الفيديو الأعلى ونركز في هذا المكان نضعه على ونذهب إلى Effect Controls ونرى ماذا يوجد هنا في هذا المكان لدينا 4 أفتحة يوجد left, top, right, bottom Crop يعني أنك تقطع حتة من الفيديو فأنت تقول أن تقطع من أنا وحتة من الشمال أو من اليمين أو من فوق أو من تحت فنقطع من الشمال قليلاً لنحس أنه قطع حتة بشكل أكيد ثم نخرج إلى فوق نذهب إلى Position ونعود إلى مكان الثاني نقوم بمسطره في نصف الفيديو وبعد ذلك يمكننا أن نقوم بعمل Select الفيديو الثاني ونذهب إلى Position ونحركه لغاية أن نظهر جزءنا من الفيديو بنفس الشكل على سبيل المثال مثلاً نجرب نشغل الفيديو نجد كل فيديو شغال في جزءه من الشاشة ونشرح كيف نقوم بعمل تأثير بسجل ونقوم بعمل خط يمكننا أن نضع نفس الخط ونقوم بعمل فيديو هنا لكي يكون شكله أفضل ونقوم بشرح الطريقة بالتفصيل في فيديو قبل هذا ستجد لنكتب في الهاتف نذهب إلى ال PEN TOOL ونضغط على سيدس TOOL ثم نذهب إلى أول فيديو نضغط على الضغط ونضغط على نقاط ونقوم برسم في آخر الفيديو نقضي نقوم بعمل خط ونقوم بشرح الشكل يمكنك أن تضعه في فيديو نقوم باختار سلكشن TOOL نقوم بشرح الفيديو ونضغط على قدر الفيديوين تكون شكله أفضل يمكنك أن تلعب في شكل خط من Effect Controls نقوم بشرح الشكل ونقوم بشرح Effect Controls تجد شيب هنا كل شيء يعلق به يوجد فيه فل وストروك يمكنك أن تغير اللون كما تريد نضغط على هذا ونختار اللون كما تريد ونقوم بشرح الأبيض يمكنك أيضا أن تضغط على الشكل هذا هو بردر الشكل يمكنك أيضا أن تضغط على بردر أبيض ونضغط على حجمه لكي يكبر الخط بالشكل كبير لكي يظهر بس أكثر ونحاول أن نشغل الفيديو الآن يكون بشكل كبير وننتظر لنهت الحلقة لا تنسى أن تشترك في القناة ونشترك في الفيسبوك وكذلك سلام',
Much better transcript this time.

``` python
model = whisper.load_model('large')
transcript_output = model.transcribe(audio)
transcript_output
```

    100%|██████████████████████████████████████| 2.87G/2.87G [00:22<00:00, 134MiB/s]

    {'text': ' كيفية تأثير الـ Split Screen على الـ Adobe Premiere Pro هيا بنا أول شيء سنفعله هو جلب فيديوهاتنا ونضعها فوق بعضنا كما نفعل وبعد ذلك نختار الفيديو الذي في الأعلى ونذهب إلى Effect Controls ونلعب في المقاومة حتى لا يبقى في نصف الشاشة بالضبط سنفعل هذا فقط ونجلبه إلى هنا يمكن أن نكون أثناء نقص الشاشة ولكن هناك مشكلة هناك جزء من الفيديو من ناحية اليمين نحن خافناه ونريده أن يظهر نريده أن يظهر على اليمين فما هي حل المشكلة؟ سأعود إلى المكان وبعد ذلك نذهب إلى Effects وهنا ونضع الشاشة على effect اسمه Drop نضعه على الفيديو الذي في الأعلى ونركز على الفيديو الذي في الأعلى نضغط عليه ونذهب إلى Effects Controls سنجد ماذا يوجد هنا بشكل هذا الآن لدينا أربع أفشل مرتبطة اليوم لفت وطوب ورايت وبوتن كروب يعني أنك تقطع حتى من الفيديو فأنت يريد أن تقطع من أي حتى من الشمال أو من اليمين أو من فوق أو من تحت سنقطع من الشمال قليلاً لكي نشعر أنه تم تصميمه بشكل هذا وبعد ذلك نذهب إلى الأعلى نذهب إلى Position ونعود إلى نسنتره في نصف الفيديو وبعد ذلك يصبح الأمر سهلاً نستطيع أن نقوم بمقاطع الفيديو الثانية ونذهب إلى Position نحركه حتى نظهر الجزء الذي نريده من الفيديو بنفس الشكل على سبيل المثال مثلاً بشكل هذا نجرب أن نشغل الفيديو نجد أن كل فيديو يعمل في جزء من الشاشة وكذلك نشرح لكم كيفية تأثير الـSplit Screen طبعاً إذا أردت أن تقوم بشيء مثل فصل أو خط يفصل من بين الفيديوهين يمكن أن نرسم خط ونضعه من بين الفيديوهين لكي يصبح شكله أفضل كنا شرحنا الطريقة بالتفصيل في الفيديو قبل ذلك ستجدون رابط في الآيفون سنذهب إلى Pen Tool سنضع سادس تول وبعد ذلك نذهب إلى الفيديو الأول إلى الوضع الذي نريده من خلال الخط نضغط على الضغطة ونضغط على الماوس ونرسم في آخر الفيديو تحت نقطة سنرسم الخط بشكل هذا ستجدون شكل الكون هنا ستجده فوق الفيديو سنختار Selection Tool سنقوم بإختيار الفيديو ونضعه على قد الفيديوهين بالضبط بشكل هذا يمكنك أيضاً أن تلعب في شكل الخط هذا من Effect Controls نستخدم Selection Tool ونذهب إلى Effect Controls ستجدون شكل الكون هنا ستجد كل شيء يتعلق به ستجد فيه Fill وStroke يمكن أن تغير اللون كما تريد يمكن أن نضغط هنا ونختار اللون الذي تريده سأختار مثال أبيض يمكن أن تضغط على Stroke Stroke هو البردر على الشكل يمكن أن نضعه على البردر أبيض ويمكن أن نزود حجمه لكي نكبر خطة الخط بشكل هذا لكي يظهر أكثر وهكذا، سنجرب ونشغل الفيديو سنجد ما يبقى بالشكل وكذلك نصل إلى نهاية الحلقة لا تنسوا إعجاب الفيديو والاشتراك في القناة وإشتركوا على صفحة الفيسبوك وهكذا، سلام',
Almost perfect, spotted only two errors, much lower WER compared to the
official 15.7

Now that we have an acceptable WER transcript, let\'s preprocess the
text segments, and first prepare the list of segments with links to
their corresponding places in the video.

``` python
transcript_data = []
for segment in transcript_output['segments']:
        transcript_data.append({
                "id": f"{youtube_video_url}&t={segment['start']}s",
                "text": segment["text"].strip(),
                "start": segment['start'],
                "end": segment['end']
        })
```

``` python
transcript_data[0:5]
```

    [{'id': 'https://www.youtube.com/watch?v=RgG5e87-Lbs&t=0.0s',
      'text': 'كيفية تأثير الـ Split Screen على الـ Adobe Premiere Pro',
      'start': 0.0,
      'end': 3.2},
     {'id': 'https://www.youtube.com/watch?v=RgG5e87-Lbs&t=3.48s',
      'text': 'هيا بنا',
      'start': 3.48,
      'end': 4.48},
     {'id': 'https://www.youtube.com/watch?v=RgG5e87-Lbs&t=11.24s',
      'text': 'أول شيء سنفعله هو جلب فيديوهاتنا ونضعها فوق بعضنا',
      'start': 11.24,
      'end': 14.84},
     {'id': 'https://www.youtube.com/watch?v=RgG5e87-Lbs&t=14.84s',
      'text': 'كما نفعل',
      'start': 14.84,
      'end': 16.080000000000002},
     {'id': 'https://www.youtube.com/watch?v=RgG5e87-Lbs&t=16.080000000000002s',
      'text': 'وبعد ذلك نختار الفيديو الذي في الأعلى',
      'start': 16.080000000000002,
      'end': 18.080000000000002}]

Combining each 5 (window) transcript segments into one, with overlapping
of 2 segments =(stride), these numbers might need fine tunning to obtain
the most meaningful segmentation

``` python
transcript_new_data = []
window = 5  # number of segments to combine
stride = 2  # number of segments to 'stride' over, used to create overlap
for i in (range(0, len(transcript_data), stride)):
    i_end = min(len(transcript_data)-1, i+window)
    text = ' '.join(_['text'] for _ in transcript_data[i:i_end])
    transcript_new_data.append({
        'start': transcript_data[i]['start'],
        'end': transcript_data[i_end]['end'],
        'text': text,
        'id': transcript_data[i]['id'],
    })
```

``` python
transcript_new_data[0:5]
```

    [{'start': 0.0,
      'end': 20.080000000000002,
      'text': 'كيفية تأثير الـ Split Screen على الـ Adobe Premiere Pro هيا بنا أول شيء سنفعله هو جلب فيديوهاتنا ونضعها فوق بعضنا كما نفعل وبعد ذلك نختار الفيديو الذي في الأعلى',
      'id': 'https://www.youtube.com/watch?v=RgG5e87-Lbs&t=0.0s'},
     {'start': 11.24,
      'end': 24.76,
      'text': 'أول شيء سنفعله هو جلب فيديوهاتنا ونضعها فوق بعضنا كما نفعل وبعد ذلك نختار الفيديو الذي في الأعلى ونذهب إلى Effect Controls ونلعب في المقاومة حتى لا يبقى في نصف الشاشة بالضبط',
      'id': 'https://www.youtube.com/watch?v=RgG5e87-Lbs&t=11.24s'},
     {'start': 16.080000000000002,
      'end': 27.44,
      'text': 'وبعد ذلك نختار الفيديو الذي في الأعلى ونذهب إلى Effect Controls ونلعب في المقاومة حتى لا يبقى في نصف الشاشة بالضبط سنفعل هذا فقط ونجلبه',
      'id': 'https://www.youtube.com/watch?v=RgG5e87-Lbs&t=16.080000000000002s'},
     {'start': 20.76,
      'end': 31.32,
      'text': 'ونلعب في المقاومة حتى لا يبقى في نصف الشاشة بالضبط سنفعل هذا فقط ونجلبه إلى هنا يمكن أن نكون أثناء نقص الشاشة',
      'id': 'https://www.youtube.com/watch?v=RgG5e87-Lbs&t=20.76s'},
     {'start': 25.16,
      'end': 35.160000000000004,
      'text': 'ونجلبه إلى هنا يمكن أن نكون أثناء نقص الشاشة ولكن هناك مشكلة هناك جزء من الفيديو من ناحية اليمين',
      'id': 'https://www.youtube.com/watch?v=RgG5e87-Lbs&t=25.16s'}]

Now, we text embed all segments


``` python
# imports
import pandas as pd
import tiktoken
from openai.embeddings_utils import get_embedding, cosine_similarity
```

``` python
# embedding model parameters
embedding_model = "text-embedding-ada-002"
embedding_encoding = "cl100k_base"  # this the encoding for text-embedding-ada-002
max_tokens = 8000  # the maximum for text-embedding-ada-002 is 8191
```

``` python
encoding = tiktoken.get_encoding("cl100k_base")
# should print [83, 1609, 5963, 374, 2294, 0]
encoding.encode("tiktoken is great!")
```

    [83, 1609, 5963, 374, 2294, 0]

``` python
import openai
df = pd.DataFrame(transcript_new_data)
openai.api_key = 'sk-PUT YOUR OPENAI API KEY'
df["embedding"] = df.text.apply(lambda x: get_embedding(x, engine=embedding_model))
```


We then embed the query, here the question is when in the video the
YouTuber asked us to subscribe to the channel.

``` python
query = 'امتى طلب اننا نعمل إشتراك فى القناه ؟'
query_embedding = get_embedding(query, engine=embedding_model)
```
We find the most relevant parts of the video transcript to the query


``` python
df["similarity"] = df.embedding.apply(lambda x: cosine_similarity(x, query_embedding))
results = (df.sort_values("similarity", ascending=False).head(3))
results = results.set_index(['id'])
```

``` python
from IPython.display import HTML
HTML(results[['start','similarity', 'text']].to_html(render_links=True, escape=False))
```


```html
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>start</th>
      <th>similarity</th>
      <th>text</th>
    </tr>
    <tr>
      <th>id</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th><a href="https://www.youtube.com/watch?v=RgG5e87-Lbs&t=190.36s" target="_blank">https://www.youtube.com/watch?v=RgG5e87-Lbs&t=190.36s</a></th>
      <td>190.36</td>
      <td>0.839812</td>
      <td>وهكذا، سنجرب ونشغل الفيديو سنجد ما يبقى بالشكل وكذلك نصل إلى نهاية الحلقة لا تنسوا إعجاب الفيديو والاشتراك في القناة وإشتركوا على صفحة الفيسبوك</td>
    </tr>
    <tr>
      <th><a href="https://www.youtube.com/watch?v=RgG5e87-Lbs&t=20.76s" target="_blank">https://www.youtube.com/watch?v=RgG5e87-Lbs&t=20.76s</a></th>
      <td>20.76</td>
      <td>0.837695</td>
      <td>ونلعب في المقاومة حتى لا يبقى في نصف الشاشة بالضبط سنفعل هذا فقط ونجلبه إلى هنا يمكن أن نكون أثناء نقص الشاشة</td>
    </tr>
    <tr>
      <th><a href="https://www.youtube.com/watch?v=RgG5e87-Lbs&t=25.16s" target="_blank">https://www.youtube.com/watch?v=RgG5e87-Lbs&t=25.16s</a></th>
      <td>25.16</td>
      <td>0.831657</td>
      <td>ونجلبه إلى هنا يمكن أن نكون أثناء نقص الشاشة ولكن هناك مشكلة هناك جزء من الفيديو من ناحية اليمين</td>
    </tr>
  </tbody>
</table>
```

References: <https://lablab.ai/t/whisper-transcribe-youtube-video>
