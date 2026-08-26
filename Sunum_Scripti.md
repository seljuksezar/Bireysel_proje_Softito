# Sunum Scripti — Sunucu Log Kayıtlarında Sistem Arızası Tespiti

**Hedef süre:** 15–17 dakika (sorular dahil değil)
**Kullanım:** Her slayt için konuşma metni ve geçiş cümleleri aşağıda. Parantez içindeki süreler o slayt için hedeflenen süredir. *(…)* ile işaretli yerlerde ekrandaki öğeye işaret edin.

---

## Slayt 1 — Başlık *(30 sn)*

> Merhaba, gününüz aydın olsun. Benim adım Selçuk Sezer. Bugün size Veri Analitiği bireysel projem olan "Sunucu Log Kayıtlarında Sistem Arızası Tespiti" çalışmamı sunacağım.
>
> Kısaca özetlemem gerekirse: yaklaşık 743 megabaytlık ham süper bilgisayar logundan, hiçbir arıza etiketi görmeden, bir sistem arızasını gerçekleşmeden önce yakalamaya çalışan bir erken uyarı sistemi geliştirdim. Nasıl yaptığımı ve yol boyunca karşılaştığım sürpriz sonuçları anlatmama izin verin.

**Geçiş:** Önce bugün nelerden bahsedeceğime bakalım.

---

## Slayt 2 — Ajanda *(30 sn)*

> Sunumum altı ana başlıktan oluşuyor. Önce problemi ve amacı tanımlayacağım; ardından kullandığım veri kümesini tanıtacağım. Daha sonra üç adımda yöntemimi anlatacağım: log ayrıştırma, kayan pencerelerle öznitelik çıkarımı ve Isolation Forest modeli. Son olarak deneysel sonuçlarımı paylaşacak, gözetimsiz modelimi denetimli derin öğrenme modelleriyle karşılaştırıp çıkarımlarımı aktaracağım.

**Geçiş:** Peki neden böyle bir çalışmaya ihtiyaç var?

---

## Slayt 3 — Problem ve Amaç *(60 sn)*

> Büyük ölçekli sistemler her gün milyonlarca konsol logu üretiyor. Bu logların içinde aslında arızaların erken sinyalleri saklı; ama bu sinyalleri elle tespit etmek imkânsız. *(sağdaki akışa işaret edin)*
>
> Projemin amacı şu: Blue Gene/L süper bilgisayarının ham loglarından hareketle, gözetimsiz öğrenme yaklaşımlarından Isolation Forest algoritmasıyla anormal davranışları tespit etmek ve arızayı ileriye dönük öngörmek.
>
> Buradaki kritik nokta şu: Geleneksel denetimli modellerin aksine benim modelim hiçbir arıza etiketi görmeden sadece "normal davranışı" öğreniyor ve normalden sapan durumları anomali sayıyor. Elimdeki gerçek alert etiketlerini ise yalnızca sonunda "acaba model gerçekten doğru şeyi yakalıyor mu?" diye doğrulama yapmak için kullandım.
>
> Yani sorumuz çok net: *(altta vurgulayın)* "Önümüzdeki çeyrek saatte bir arıza olacak mı?"

---

## Slayt 4 — Proje Kısıtları *(45 sn)*

> Çalışmanın kapsamını üç kısıtla netleştirdim.
>
> Birincisi odak kısıtı: Ana odağım Isolation Forest tabanlı gözetimsiz tespit ve onun gerektirdiği veri hazırlama adımlarıydı. Denetimli modelleri ise yalnızca karşılaştırma amaçlı, özet sonuçlar düzeyinde dahil ettim; ayrıntılı mimari optimizasyon kapsam dışında kaldı.
>
> İkincisi veri kısıtı: Yalnızca Los Alamos Ulusal Laboratuvarı'nın paylaştığı BGL veri kümesiyle çalıştım; başka sistemlere genelleme yapmadım.
>
> Üçüncüsü de tespit düzeyiyle ilgili: Tahminler pencere bazında yapıldı. Tekil satır düzeyi tespit, gerçek zamanlı streaming senaryoları ve dağıtık eğitim kapsam dışında.

**Geçiş:** Şimdi verimize yakından bakalım.

---

## Slayt 5 — BGL Veri Kümesi *(60 sn)*

> Kullandığım BGL veri kümesi, Los Alamos Ulusal Laboratuvarı'ndaki 131 bin işlemcili — ki bu onu dünyanın en hızlı süper bilgisayarlarından biri yapıyor — Blue Gene/L sisteminin konsol loglarından oluşuyor. Log tabanlı arıza tahmini araştırmalarında standart bir kıyaslama kümesi sayılıyor.
>
> Rakamlarla: *(kartlara işaret edin)* Ham dosya 743 megabayt; ayrıştırma sonrası üç milyon on yedi bin log satırı elde ediyoruz; beş bin farklı olay tipi var.
>
> Etiketleme mantığı şöyle: `KERN`, `APP`, `RAS` gibi öneklerle başlayan satırlar alert — yani gerçek bir arıza ya da uyarı durumu — olarak işaretlenmiş. Alert oranı satır bazında yüzde sekiz virgül dört. Bu bile başlı başına bir dengesizlik; ama birazdan göreceğiniz gibi pencere düzeyine geçtiğimizde bu oran çok daha dramatik biçimde düşüyor.

---

## Slayt 6 — Uçtan Uca Pipeline *(60 sn)*

> Çalışmanın tamamı tek bir hat üzerinde ilerliyor. *(chevron akışını soldan sağa takip edin)*
>
> Ham loglar önce Drain algoritmasıyla yapılandırılmış olay şablonlarına dönüşüyor. Ardından veri on beş dakikalık örtüşmesiz pencerelere bölünüyor; toplam beş bin dokuz yüz elli yedi pencere ortaya çıkıyor. Her pencereden beş bin sekiz boyutlu bir öznitelik vektörü çıkarılıyor, standardize ediliyor ve Isolation Forest'a veriliyor. Model iki yüz ağaçla, contamination değeri 0,06 olacak şekilde eğitildi. En sonda da gerçek alert etiketleriyle sanity check yapıyoruz.
>
> Alt tarafta gördüğünüz gibi bu hattın tüm ara ürünleri diskte duruyor: parsed.csv dosyası, öznitelik matrisi ve eğitilmiş model. Tüm parametreler tek bir config dosyasından yönetiliyor; yani çalışma tamamen tekrarlanabilir.

**Geçiş:** Hattın ilk adımını açalım.

---

## Slayt 7 — Adım 1: Drain ile Log Ayrıştırma *(75 sn)*

> Ham log satırları yapısal olmayan düz metinler; bunları doğrudan makine öğrenmesine veremezsiniz. Önce her satırın sabit bir mesaj şablonuna dönüştürülmesi gerekiyor.
>
> Bunun için Drain algoritmasını tercih ettim. Drain çevrimiçi çalışan, hesaplama verimliliği yüksek bir ayrıştırıcı: log satırlarını önce uzunluklarına göre grupluyor, sonra sabit derinlikli bir ağaç yapısında token benzerliğine göre yönlendiriyor. Birbirine benzeyen her gruba benzersiz bir event ID atanıyor. Benim verimde bu, beş bin farklı olay tipine karşılık geldi.
>
> Sol alttaki örnek dönüşüme bakarsanız: "cpu 3 temperature 87C exceeds threshold" gibi bir satır, "cpu ID temperature N exceeds threshold" şablonuna indirgeniyor. Yani değişkenler soyutlanıyor, anlam korunuyor.
>
> Sağda kullandığım parametreleri görüyorsunuz: benzerlik eşiği 0,5, ağaç derinliği 4, düğüm başına en fazla 100 çocuk, mesaj başına en fazla 128 token.
>
> Çıktımız parsed.csv dosyası: zaman damgası, ikili alert etiketi, event ID ve şablon metni. Artık veri modellemeye hazır.

---

## Slayt 8 — Adım 2: Kayan Pencere ve Öznitelikler *(75 sn)*

> İkinci adım zamansal yapıyı modele taşımak. Log verisi doğası gereği bir zaman serisi olduğu için veriyi on beş dakikalık, örtüşmesiz pencerelere böldüm ve her pencereyi tek bir gözlem olarak ele aldım.
>
> Etiketleme mantığı ekranın altında: *(banda işaret edin)* Bir pencerenin kapanışını izleyen on beş dakika içinde en az bir alert gerçekleştiyse o pencere pozitif, değilse negatif. Model böylece "önümüzdeki çeyrek saatte arıza olacak mı?" sorusuna yanıt veriyor.
>
> Peki her pencereden ne çıkardım? *(sağdaki kartları takip edin)* Dört grup öznitelik var: Birincisi olay frekans histogramı — her olay tipinin penceredeki göreli sıklığı, beş bin ikinci boyutlu bir vektör. İkincisi meta öznitelikler: alert oranı, benzersiz olay sayısı, sekans uzunluğu. Üçüncüsü yoğunlaşma ölçüleri: en sık on olayın toplam frekansa oranı ve maksimum tek olay frekansı. Dördüncüsü Shannon entropisi — olay dağılımının belirsizliğini ölçüyor.
>
> Hepsi birleşince pencere başına beş bin sekiz boyutlu bir vektör elde ediyoruz.

---

## Slayt 9 — Sınıf Dengesizliği *(60 sn)*

> Bu slayt, çalışmanın belki de en belirleyici gerçeğini gösteriyor. *(halka grafiğe işaret edin)*
>
> Toplam beş bin dokuz yüz elli yedi pencerede pozitiflerin oranı yalnızca yüzde dört virgül kırk sekiz — iki yüz altmış yedi pencere. Daha da ilginci test bölmesinde: bin yüz doksan iki pencereden yalnızca yirmi ikisi pozitif; oran yüzde bir virgün seksen beşe düşüyor.
>
> Lütfen bu rakamı aklınızda tutun, çünkü hem gözetimsiz modelimin precision değerini hem de denetimli modellerin başarısızlığını açıklayan ana etken bu.
>
> Pozitif örnekler bu kadar nadirken hem değerlendirme hem de model seçimi stratejik hâle geliyor.

---

## Slayt 10 — Adım 3: Isolation Forest *(75 sn)*

> Ve modelimizin kendisi. Isolation Forest'ın fikri çok zarif: Anomaliler zaten azınlıktadır ve diğer gözlemlerden uzaktadır. Dolayısıyla rastgele bölmelerle **daha az adımda izole edilirler**. Kısa yol uzunluğu, anomali işaretidir.
>
> Algoritma, rastgele seçilen öznitelik ve bölme değerleriyle yüzlerce ikili izolasyon ağacı kuruyor; bir gözlemin anomali skorunu ağaçlar arası ortalama yol uzunluğundan hesaplıyor. Formülde *(formül kutusuna işaret edin)* skor 1'e yaklaşırsa anomali, 0,5'in altına inerse normal kabul ediliyor. Contamination parametresini 0,06 alarak karar eşiğini veride beklenen anomali oranına göre konumlandırdım.
>
> Peki neden tam olarak bu algoritma? *(sağdaki listeden okuyun)* Dört gerekçe: Birincisi ve en önemlisi etiket gerektirmemesi — gerçek hayatta arıza etiketli veri çoğu zaman yoktur, bu model etiketsiz yeni sistemlere taşınabilir. İkincisi yüksek boyutlu seyrek vektörlerde verimli olması. Üçüncüsü doğrusal olmayan ilişkileri yakalayabilmesi. Dördüncüsü paralel hesaplamaya elverişliliği.

---

## Slayt 11 — Eğitim Kurulumu *(50 sn)*

> Eğitim tarafında üç kritik kararı vurgulamak istiyorum.
>
> İlki veri bölmeleme: Zaman serisiyle çalıştığımız için ayrımı kronolojik yaptım — karıştırmayı bilinçli olarak uygulamadım. İlk yüzde seksen eğitim, son yüzde yirmi test.
>
> İkincisi ölçeklendirme: Frekans değerleri sıfır ile bir arasında kalırken olay sayıları yüzlerce mertebesinde; StandardScaler ile standardize ettim. Dikkat edilecek nokta şu: Ölçekleyiciyi **yalnızca eğitim bölmesinde** fit ettim, aynı dönüşümü test'e uyguladım. Bilgi sızıntısı yok.
>
> Üçüncüsü çıktı dönüşümü: scikit-learn'ün predict fonksiyonu anomalileri eksi bir ile işaretliyor; bunları bir'e, normalleri sıfıra çevirerek gerçek etiketlerle karşılaştırılabilir hâle getirdim. Hiperparametreler sağdaki tabloda: iki yüz ağaç, contamination 0,06, seed 42 ile tekrarlanabilirlik garanti altında.

---

## Slayt 12 — Deneysel Sonuçlar *(75 sn)*

> İşte sonuçlar. Test bölmesinde bin yüz doksan iki pencere vardı, yirmi ikisi gerçekten pozitifti.
>
> Genel doğruluk yüzde doksan yedi virgül altı görünüyor; ama dürüst olmak gerekirse bunun büyük kısmı çoğunluk sınıfından geliyor. Asıl hikâyeyi sağdaki sayılar anlatıyor.
>
> Precision 0,24, recall 0,27, F1 0,26. Ama benim için asıl başarı göstergesi burada: *(gauge'a işaret edin)* ROC-AUC 0,81. Bu ne demek? Hiçbir arıza etiketi görmemiş bir modelin anomali skorları, gerçek pozitif pencereleri normal pencerelerden belirgin biçimde ayırt edebiliyor demek.
>
> Düşük precision'ı bağlamıyla okumak lazım: Model "arıza kesin gerçekleşecek pencere"yi değil, "istatistiksel olarak sıra dışı pencere"yi işaret ediyor; üstelik pozitif oran yüzde bir virgül seksen beş gibi aşırı düşükken bu beklenen bir sonuç.
>
> Ve erken uyarı senaryosunda kritik metrik recall'dur: Bir arızayı kaçırmak, yanlış alarm üretmekten çok daha pahalıdır. Recall'u nasıl yönetebileceğimizi bir sonraki slaytta göreceğiz.

---

## Slayt 13 — Eşik ve Pencere Analizi *(75 sn)*

> Model eğitmekle iş bitmiyor; operasyonel ayar en az model kadar önemli. İki analiz yaptım.
>
> Solda eşik süpürme: Varsayılan eşikte model yirmi beş alarm üretiyor ve gerçek anomalilerin yüzde yirmi yedini yakalıyor. Eşiği hafifçe düşürdüğümüzde — 0,2940'dan 0,2938'e, kulağa önemsiz gelen bir değişiklikle — alarm sayısı önce altı yüz altmış ikiye çıkarken recall yüzde 91'e fırlıyor; tamamen açtığımızda yüzde yüze ulaşıyor.
>
> Buradan çıkan ders şu: Aynı model, eşik tercihiyle "hassas ama gürültülü" ya da "seçici ama kaçırmacı" bir sisteme dönüşebiliyor. Bu tercih tamamen operasyonel bir karardır.
>
> Sağda pencere genişliği deneyi: Otuz dakikalık pencerelerde recall yüzde 57,7 iken on beş dakikalık pencerelerde yüzde 27,3'e geriledi; ROC-AUC ise hafifçe arttı. Kısa pencereler anomaliyi dar bir ufka sıkıştırarak skoru daha ayırt edici kılıyor ama pozitif örnek sayısı azaldığı için duyarlılık düşüyor. Yani klasik bir ödünleşim var ve pencere genişliği uygulama hedefine göre seçilmeli.

---

## Slayt 14 — Denetimli Modellerle Karşılaştırma *(75 sn)*

> Şimdi sunumun en ilginç bölümüne geldik: plot twist.
>
> Karşılaştırma amacıyla literatürün güçlü mimarilerinden LogRobust'u — BiLSTM ve multi-head attention katmanlı bir modeli — aynı veri üzerinde üç farklı kayıp fonksiyonuyla eğittim. Sonuçlar tabloda: *(tabloyu takip edin)* CrossEntropy ile ROC-AUC 0,44. BCE ile 0,71. Focal Loss ile 0,37 — rastgele tahminin bile altında.
>
> Ama asıl çarpıcı olan şu: Üç konfigürasyonun **tamamında** precision ve F1 sıfır. Yani bu modeller karar eşiğinde tek bir pozitif pencereyi bile isabetle seçemedi. Grafiğin hikâyesi net: *(grafiğe işaret edin)* kırmızılar denetimli modeller, yeşil bizim gözetimsiz Isolation Forest — hem sıralamada hem kullanılabilirlikte açık ara önde.
>
> En iyi BiLSTM bile BCE ile ancak 0,71 sıralama başarımı gösterdi; ama bu başarı eşiğe hiç dönüşmedi.

---

## Slayt 15 — Neden Başarısız Oldu? *(60 sn)*

> Peki güçlü bir mimari neden bu kadar kötü performans gösterdi? Üç birleşik neden buldum.
>
> Bir: Aşırı sınıf dengesizliği. Test bölmesinde pozitif oran yüzde bir virgül seksen beş. Derin modeller bu kadar az pozitif örnekle çoğunluk sınıfını tahmin etmeyi "öğrenmeye" yöneliyor.
>
> İki: Parametre hacmi ile veri hacmi uyumsuzluğu. Multi-head attention'lı bir BiLSTM'in milyonlarca parametresi var; dört bin yedi yüz altmış beş pencerelik eğitim kümesi bu mimariyi besleyemiyor. Overfitting ya da underfitting kaçınılmaz.
>
> Üç: Kayıp fonksiyonu kalibrasyonu. Standart BCE'de gradyan çoğunluk sınıfınca baskılanıyor; Focal Loss ise gamma ve alpha parametrelerinin ince ayarına muhtaç — ayarlanamadığında AUC'nin 0,37'ye düşmesi tam da bunun kanıtı.
>
> Alttaki nüans önemli: Bu başarısızlığın sebebi denetimli yöntemin doğası değil; veri hacmi, dengesizlik ve kalibrasyon koşulları. SMOTE, ağırlıklı örnekleme ve iyi bir eşik kalibrasyonuyla bu modellerin rövanş potansiyeli var.

---

## Slayt 16 — Sonuç *(60 sn)*

> Toparlarsam dört ana çıkarımım var.
>
> Birincisi: Uçtan uca, tekrarlanabilir bir gözetimsiz hat kurdum — Drain'den Isolation Forest'a, seed sabitlenmiş hâlde.
>
> İkincisi ve bence en değerlisi: Etiketsiz öğrenme gerçekten çalışıyor. Hiçbir arıza etiketi görmeden ROC-AUC 0,81 ile gerçek alert pencerelerinin önemli bölümünü yakaladık. Etiket maliyeti sıfırken anlamlı bir erken uyarı sinyali üretmek, pratikte büyük değer.
>
> Üçüncüsü: Bu veri koşullarında gözetimsiz yaklaşım, denetimli alternatiflerden daha güvenilir bir temel oluşturdu.
>
> Ve dördüncüsü: Eşik ayarı operasyonel esneklik sağlıyor; pencere genişliği ise hedefe göre verilmesi gereken kritik bir tasarım kararı.

---

## Slayt 17 — Gelecek Çalışmalar *(45 sn)*

> Bu çalışmayı dört yönde geliştirmeyi planlıyorum.
>
> Birincisi contamination parametresini sabit tutmak yerine, zaman içindeki anomali oranına göre dinamik hâle getirmek. İkincisi meta özniteliklere pencereler arası trend bilgisini eklemek — ani yükselişler muhtemelen tekil pencere istatistiklerinden daha bilgi taşıyor. Üçüncüsü denetimli modellerin rövanşı: SMOTE ya da ağırlıklı örnekleme ve düzgün eşik kalibrasyonuyla BiLSTM'leri yeniden değerlendirmek. Ve dördüncüsü, gözetimsiz modelle kalibre edilmiş denetimli modeli hibrit bir ensemble'da birleştirmek.

---

## Slayt 18 — Teşekkürler *(20 sn)*

> Dinlediğiniz için teşekkür ederim. Sorularınızı memnuniyetle yanıtlarım.

---

# Olası Sorulara Hazırlık (Q&A)

| Olası soru | Kısa yanıt |
|---|---|
| **Precision 0,24 ile bu sistem üretime girer mi?** | Tek başına hayır; ama eşik ayarıyla recall %91+ seviyesine çekilebilir. Erken uyarı senaryosunda amaç zaten yüksek recall'dur; alarm triyajı operatör katmanında çözülür. |
| **Neden LogBERT/XGBoost denemediniz?** | Proje kısıtı gereği odak gözetimsiz hat + özet düzeyde denetimli karşılaştırmaydı. Gelecek çalışmalar maddesinde kalibre edilmiş denetimli modellerin yeniden değerlendirilmesi var. |
| **Alert etiketi kullandınız, "gözetimsiz" nasıl oluyor?** | Etiketler modele hiçbir aşamada verilmedi — ne eğitimde ne ölçeklendirmede. Yalnızca pencere etiketlemesi (değerlendirme için) ve son doğrulamada kullanıldılar. |
| **Pencere neden 15 dakika?** | Operasyonel bir varsayım; horizon da aynı. Deney, 30 dk'ya kıyasla trade-off'u gösterdi: kısa pencere AUC'yi koruyor ama recall'u düşürüyor. Hedefe göre ayarlanmalı. |
| **Contamination 0,06'yı nasıl seçtiniz?** | Satır bazlı alert oranı %8,4; pencere bazında %4,5 civarı. 0,06 bu ikisinin arasında makul bir orta nokta; dinamik belirlenmesi gelecek çalışma. |
| **Drain yerine regex/neden başka parser olmadı?** | BGL'de binlerce farklı mesaj varyantı var; manuel regex sürdürülemez. Drain çevrimiçi, hızlı ve benchmark çalışmalarında standart. |
| **Sonuç genelenebilir mi?** | Tek sistem, tek dönem. Ancak yöntemin kendisi (parse → window → features → IF) sistemden bağımsız; farklı log kümelerinde aynı hat kurulabilir. |

---

## Sunum İpuçları

- **Toplam ~15-16 dk.** Süre daralırsa kısaltılacak yerler: Slayt 4 (kısıtları tek cümleye indirin) ve Slayt 11 (sadece kronolojik split + scaler vurgusu).
- **En güçlü anlar:** Slayt 12'deki gauge (0,81) ve Slayt 14'teki "plot twist". Bu iki slaytta tempoyu düşürün, izleyiciye sayıları okutun.
- **Slayt 9'un rakamını (%1,85)** ezberden söyleyin — Sl. 15'teki açıklamaya köprü kurar.
- Terminal ekran görüntülerini Slayt 6 ve 12'ye eklediyseniz, "gerçekten çalıştırılmış bir sistem" algısını pekiştirmek için birer cümleyle atıfta bulunun.
